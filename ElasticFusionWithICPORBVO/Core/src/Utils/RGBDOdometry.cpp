/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#include "RGBDOdometry.h"
#include <numeric>
#include <cstdlib> /* 亂數相關函數 */
#include <ctime>   /* 時間相關函數 */
RGBDOdometry::RGBDOdometry(int width,
                           int height,
                           float cx, float cy, float fx, float fy,
                           float distThresh,
                           float angleThresh)
: lastICPError(0),
  lastICPCount(width * height),
  lastRGBError(0),
  lastRGBCount(width * height),
  lastSO3Error(0),
  lastSO3Count(width * height),
  lastA(Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero()),
  lastb(Eigen::Matrix<double, 6, 1>::Zero()),
  sobelSize(3),
  sobelScale(1.0 / pow(2.0, sobelSize)),
  maxDepthDeltaRGB(0.07),
  maxDepthRGB(6.0),
  distThres_(distThresh),
  angleThres_(angleThresh),
  width(width),
  height(height),
  cx(cx), cy(cy), fx(fx), fy(fy)
{
    sumDataSE3.create(MAX_THREADS);
    outDataSE3.create(1);
    sumResidualRGB.create(MAX_THREADS);

    sumDataSO3.create(MAX_THREADS);
    outDataSO3.create(1);

    for(int i = 0; i < NUM_PYRS; i++)
    {
        int2 nextDim = {height >> i, width >> i};
        pyrDims.push_back(nextDim);
    }

    for(int i = 0; i < NUM_PYRS; i++)
    {
        lastDepth[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
        lastImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        nextDepth[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
        nextImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        lastNextImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        nextdIdx[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
        nextdIdy[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        pointClouds[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        corresImg[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
    }

    intr.cx = cx;
    intr.cy = cy;
    intr.fx = fx;
    intr.fy = fy;

    iterations.resize(NUM_PYRS);

    depth_tmp.resize(NUM_PYRS);

    vmaps_g_prev_.resize(NUM_PYRS);
    nmaps_g_prev_.resize(NUM_PYRS);

    vmaps_curr_.resize(NUM_PYRS);
    nmaps_curr_.resize(NUM_PYRS);

    for (int i = 0; i < NUM_PYRS; ++i)
    {
        int pyr_rows = height >> i;
        int pyr_cols = width >> i;

        depth_tmp[i].create (pyr_rows, pyr_cols);

        vmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);
        nmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);

        vmaps_curr_[i].create (pyr_rows*3, pyr_cols);
        nmaps_curr_[i].create (pyr_rows*3, pyr_cols);
    }

    vmaps_tmp.create(height * 4 * width);
    nmaps_tmp.create(height * 4 * width);

    minimumGradientMagnitudes.resize(NUM_PYRS);
    minimumGradientMagnitudes[0] = 5;
    minimumGradientMagnitudes[1] = 3;
    minimumGradientMagnitudes[2] = 1;
}

RGBDOdometry::~RGBDOdometry()
{

}

void RGBDOdometry::initICP(GPUTexture * filteredDepth, const float depthCutoff)
{
    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &filteredDepth->cudaRes);

    cudaGraphicsSubResourceGetMappedArray(&textPtr, filteredDepth->cudaRes, 0, 0);

    cudaMemcpy2DFromArray(depth_tmp[0].ptr(0), depth_tmp[0].step(), textPtr, 0, 0, depth_tmp[0].colsBytes(), depth_tmp[0].rows(), cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &filteredDepth->cudaRes);

    for(int i = 1; i < NUM_PYRS; ++i)
    {
        pyrDown(depth_tmp[i - 1], depth_tmp[i]);
    }

    for(int i = 0; i < NUM_PYRS; ++i)
    {
        createVMap(intr(i), depth_tmp[i], vmaps_curr_[i], depthCutoff);
        createNMap(vmaps_curr_[i], nmaps_curr_[i]);
    }

    cudaDeviceSynchronize();
}

void RGBDOdometry::initICP(GPUTexture * predictedVertices, GPUTexture * predictedNormals, const float depthCutoff)
{
    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &predictedVertices->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedVertices->cudaRes, 0, 0);
    cudaMemcpyFromArray(vmaps_tmp.ptr(), textPtr, 0, 0, vmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &predictedVertices->cudaRes);

    cudaGraphicsMapResources(1, &predictedNormals->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedNormals->cudaRes, 0, 0);
    cudaMemcpyFromArray(nmaps_tmp.ptr(), textPtr, 0, 0, nmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &predictedNormals->cudaRes);

    copyMaps(vmaps_tmp, nmaps_tmp, vmaps_curr_[0], nmaps_curr_[0]);

    for(int i = 1; i < NUM_PYRS; ++i)
    {
        resizeVMap(vmaps_curr_[i - 1], vmaps_curr_[i]);
        resizeNMap(nmaps_curr_[i - 1], nmaps_curr_[i]);
    }

    cudaDeviceSynchronize();
}

void RGBDOdometry::initICPModel(GPUTexture * predictedVertices,
                                GPUTexture * predictedNormals,
                                const float depthCutoff,
                                const Eigen::Matrix4f & modelPose)
{
    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &predictedVertices->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedVertices->cudaRes, 0, 0);
    cudaMemcpyFromArray(vmaps_tmp.ptr(), textPtr, 0, 0, vmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &predictedVertices->cudaRes);

    cudaGraphicsMapResources(1, &predictedNormals->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedNormals->cudaRes, 0, 0);
    cudaMemcpyFromArray(nmaps_tmp.ptr(), textPtr, 0, 0, nmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &predictedNormals->cudaRes);

    copyMaps(vmaps_tmp, nmaps_tmp, vmaps_g_prev_[0], nmaps_g_prev_[0]);

    for(int i = 1; i < NUM_PYRS; ++i)
    {
        resizeVMap(vmaps_g_prev_[i - 1], vmaps_g_prev_[i]);
        resizeNMap(nmaps_g_prev_[i - 1], nmaps_g_prev_[i]);
    }

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcam = modelPose.topLeftCorner(3, 3);
    Eigen::Vector3f tcam = modelPose.topRightCorner(3, 1);

    mat33 device_Rcam = Rcam;
    float3 device_tcam = *reinterpret_cast<float3*>(tcam.data());

    for(int i = 0; i < NUM_PYRS; ++i)
    {
        tranformMaps(vmaps_g_prev_[i], nmaps_g_prev_[i], device_Rcam, device_tcam, vmaps_g_prev_[i], nmaps_g_prev_[i]);
    }

    cudaDeviceSynchronize();
}

void RGBDOdometry::populateRGBDData(GPUTexture * rgb,
                                    DeviceArray2D<float> * destDepths,
                                    DeviceArray2D<unsigned char> * destImages)
{
    verticesToDepth(vmaps_tmp, destDepths[0], maxDepthRGB);

    for(int i = 0; i + 1 < NUM_PYRS; i++)
    {
        pyrDownGaussF(destDepths[i], destDepths[i + 1]);
    }

    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &rgb->cudaRes);

    cudaGraphicsSubResourceGetMappedArray(&textPtr, rgb->cudaRes, 0, 0);

    imageBGRToIntensity(textPtr, destImages[0]);

    cudaGraphicsUnmapResources(1, &rgb->cudaRes);

    for(int i = 0; i + 1 < NUM_PYRS; i++)
    {
        pyrDownUcharGauss(destImages[i], destImages[i + 1]);
    }

    cudaDeviceSynchronize();
}

void RGBDOdometry::initRGBModel(GPUTexture * rgb)
{
    //NOTE: This depends on vmaps_tmp containing the corresponding depth from initICPModel
    populateRGBDData(rgb, &lastDepth[0], &lastImage[0]);
}

void RGBDOdometry::initRGB(GPUTexture * rgb)
{
    //NOTE: This depends on vmaps_tmp containing the corresponding depth from initICP
    populateRGBDData(rgb, &nextDepth[0], &nextImage[0]);
}

void RGBDOdometry::initFirstRGB(GPUTexture * rgb)
{
    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &rgb->cudaRes);

    cudaGraphicsSubResourceGetMappedArray(&textPtr, rgb->cudaRes, 0, 0);

    imageBGRToIntensity(textPtr, lastNextImage[0]);

    cudaGraphicsUnmapResources(1, &rgb->cudaRes);

    for(int i = 0; i + 1 < NUM_PYRS; i++)
    {
        pyrDownUcharGauss(lastNextImage[i], lastNextImage[i + 1]);
    }
}

void RGBDOdometry::getIncrementalTransformation(Eigen::Vector3f & trans,
                                                Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot,
                                                const bool & rgbOnly,
                                                const float & icpWeight,
                                                const bool & pyramid,
                                                const bool & fastOdom,
                                                const bool & so3)
{
    bool icp = !rgbOnly && icpWeight > 0;
    bool rgb = rgbOnly || icpWeight < 100;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev = rot;
    Eigen::Vector3f tprev = trans;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr = Rprev;
    Eigen::Vector3f tcurr = tprev;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr_inc;
    Eigen::Vector3f tcurr_inc;
    if(rgb)
    {
        for(int i = 0; i < NUM_PYRS; i++)
        {
            computeDerivativeImages(nextImage[i], nextdIdx[i], nextdIdy[i]);
        }
    }

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> resultR = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();

    if(so3)
    {
        int pyramidLevel = 2;

        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R_lr = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>::Identity();

        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();

        K(0, 0) = intr(pyramidLevel).fx;
        K(1, 1) = intr(pyramidLevel).fy;
        K(0, 2) = intr(pyramidLevel).cx;
        K(1, 2) = intr(pyramidLevel).cy;
        K(2, 2) = 1;

        float lastError = std::numeric_limits<float>::max() / 2;
        float lastCount = std::numeric_limits<float>::max() / 2;

        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> lastResultR = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();

        for(int i = 0; i < 10; i++)
        {
            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> jtj;
            Eigen::Matrix<float, 3, 1> jtr;

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> homography = K * resultR * K.inverse();

            mat33 imageBasis;
            memcpy(&imageBasis.data[0], homography.cast<float>().eval().data(), sizeof(mat33));

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_inv = K.inverse();
            mat33 kinv;
            memcpy(&kinv.data[0], K_inv.cast<float>().eval().data(), sizeof(mat33));

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_R_lr = K * resultR;
            mat33 krlr;
            memcpy(&krlr.data[0], K_R_lr.cast<float>().eval().data(), sizeof(mat33));

            float residual[2];

            TICK("so3Step");
            so3Step(lastNextImage[pyramidLevel],
                    nextImage[pyramidLevel],
                    imageBasis,
                    kinv,
                    krlr,
                    sumDataSO3,
                    outDataSO3,
                    jtj.data(),
                    jtr.data(),
                    &residual[0],
                    GPUConfig::getInstance().so3StepThreads,
                    GPUConfig::getInstance().so3StepBlocks);
            TOCK("so3Step");

            lastSO3Error = sqrt(residual[0]) / residual[1];
            lastSO3Count = residual[1];

            //Converged
            if(lastSO3Error < lastError && lastCount == lastSO3Count)
            {
                break;
            }
            else if(lastSO3Error > lastError + 0.001) //Diverging
            {
                lastSO3Error = lastError;
                lastSO3Count = lastCount;
                resultR = lastResultR;
                break;
            }

            lastError = lastSO3Error;
            lastCount = lastSO3Count;
            lastResultR = resultR;

            Eigen::Vector3f delta = jtj.ldlt().solve(jtr);

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rotUpdate = OdometryProvider::rodrigues(delta.cast<double>());

            R_lr = rotUpdate.cast<float>() * R_lr;

            for(int x = 0; x < 3; x++)
            {
                for(int y = 0; y < 3; y++)
                {
                    resultR(x, y) = R_lr(x, y);
                }
            }
        }
    }

    iterations[0] = fastOdom ? 3 : 10;
    iterations[1] = pyramid ? 5 : 0;
    iterations[2] = pyramid ? 4 : 0;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev_inv = Rprev.inverse();
    mat33 device_Rprev_inv = Rprev_inv;
    float3 device_tprev = *reinterpret_cast<float3*>(tprev.data());

    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> resultRt = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();

    if(so3)
    {
        for(int x = 0; x < 3; x++)
        {
            for(int y = 0; y < 3; y++)
            {
                resultRt(x, y) = resultR(x, y);
            }
        }
    }

    for(int i = NUM_PYRS - 1; i >= 0; i--)
    {
        if(rgb)
        {
            projectToPointCloud(lastDepth[i], pointClouds[i], intr, i);
        }

        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();

        K(0, 0) = intr(i).fx;
        K(1, 1) = intr(i).fy;
        K(0, 2) = intr(i).cx;
        K(1, 2) = intr(i).cy;
        K(2, 2) = 1;

        lastRGBError = std::numeric_limits<float>::max();

        for(int j = 0; j < iterations[i]; j++)
        {
            Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Rt = resultRt.inverse();

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = Rt.topLeftCorner(3, 3);

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> KRK_inv = K * R * K.inverse();
            mat33 krkInv;
            memcpy(&krkInv.data[0], KRK_inv.cast<float>().eval().data(), sizeof(mat33));

            Eigen::Vector3d Kt = Rt.topRightCorner(3, 1);
            Kt = K * Kt;
            float3 kt = {(float)Kt(0), (float)Kt(1), (float)Kt(2)};

            int sigma = 0;
            int rgbSize = 0;

            if(rgb)
            {
                TICK("computeRgbResidual");
                computeRgbResidual(pow(minimumGradientMagnitudes[i], 2.0) / pow(sobelScale, 2.0),
                                   nextdIdx[i],
                                   nextdIdy[i],
                                   lastDepth[i],
                                   nextDepth[i],
                                   lastImage[i],
                                   nextImage[i],
                                   corresImg[i],
                                   sumResidualRGB,
                                   maxDepthDeltaRGB,
                                   kt,
                                   krkInv,
                                   sigma,
                                   rgbSize,
                                   GPUConfig::getInstance().rgbResThreads,
                                   GPUConfig::getInstance().rgbResBlocks);
                TOCK("computeRgbResidual");
            }

            float sigmaVal = std::sqrt((float)sigma / rgbSize == 0 ? 1 : rgbSize);
            float rgbError = std::sqrt(sigma) / (rgbSize == 0 ? 1 : rgbSize);

            if(rgbOnly && rgbError > lastRGBError)
            {
                break;
            }

            lastRGBError = rgbError;
            lastRGBCount = rgbSize;

            if(rgbOnly)
            {
                sigmaVal = -1; //Signals the internal optimisation to weight evenly
            }
            mat33 device_Rcurr = Rcurr;
            float3 device_tcurr = *reinterpret_cast<float3*>(tcurr.data());

            DeviceArray2D<float>& vmap_curr = vmaps_curr_[i];
            DeviceArray2D<float>& nmap_curr = nmaps_curr_[i];

            DeviceArray2D<float>& vmap_g_prev = vmaps_g_prev_[i];
            DeviceArray2D<float>& nmap_g_prev = nmaps_g_prev_[i];

            //icp step start
            int sample_num;
            bool samp_flag;
            bool pyrmid_flag = true;
            if(pyrmid_flag){
                /*
                if (i == 0 || j != 0){
                    sample_num = 1;
                    samp_flag = false;
                    std::cout<<"No Rand Sample for "<< vmap_curr.cols() <<" x "<<vmap_curr.rows()/3<<"\n";
                }
                else{
                    sample_num = 10;
                    samp_flag = true;
                }
                */
                if (i == 2 && (j == 0)){
                    sample_num = 800;
                    samp_flag = true;
                }
                else{
                    sample_num = 1;
                    samp_flag = false;
                    std::cout<<"No Rand Sample for "<< vmap_curr.cols() <<" x "<<vmap_curr.rows()/3<<"\n";
                }

            }
            else{
                sample_num = 10;
                samp_flag = true;
            }
            Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_rgbd;
            Eigen::Matrix<float, 6, 1> b_rgbd;

            Eigen::Vector3f euler_angles[sample_num];
            Eigen::Vector3f trans[sample_num];
            float inlier_ratio_all[sample_num];

            if(rgb)
            {
                TICK("rgbStep");
                rgbStep(corresImg[i],
                        sigmaVal,
                        pointClouds[i],
                        intr(i).fx,
                        intr(i).fy,
                        nextdIdx[i],
                        nextdIdy[i],
                        sobelScale,
                        sumDataSE3,
                        outDataSE3,
                        A_rgbd.data(),
                        b_rgbd.data(),
                        GPUConfig::getInstance().rgbStepThreads,
                        GPUConfig::getInstance().rgbStepBlocks);
                TOCK("rgbStep");
            }

            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_rgbd = A_rgbd.cast<double>();
            Eigen::Matrix<double, 6, 1> db_rgbd = b_rgbd.cast<double>();
            Eigen::Matrix<double, 6, 1> result;
            Eigen::Matrix<double, 6, 1> result_all[sample_num];
            Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_icp = Eigen::Matrix<float, 6, 6, Eigen::RowMajor>::Zero();
            Eigen::Matrix<float, 6, 1> b_icp = Eigen::Matrix<float, 6, 1>::Zero();
            srand( time(NULL) );
            for (int k = 0; k < sample_num; k++){
                sumDataSE3cp = sumDataSE3;
                outDataSE3cp = outDataSE3;
                Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_icp = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero();
                Eigen::Matrix<double, 6, 1> db_icp = Eigen::Matrix<double, 6, 1>::Zero();
                A_icp = Eigen::Matrix<float, 6, 6, Eigen::RowMajor>::Zero();
                b_icp = Eigen::Matrix<float, 6, 1>::Zero();
                int sampled_count;
                float residual[2] = {0,0};
                if(icp)
                {
                    TICK("icpStep");
                    if(samp_flag){
                        float inlier_check[2] = {0,0};
                        char corres_map[vmap_curr.cols()*vmap_curr.rows()/3]= {0};
                        char corres_map_out[vmap_curr.cols()*vmap_curr.rows()/3] = {0};
                        GetCorresStepFull(  
                                    device_Rcurr,
                                    device_tcurr,
                                    vmap_curr,
                                    nmap_curr,
                                    device_Rprev_inv,
                                    device_tprev,
                                    intr(i),
                                    vmap_g_prev,
                                    nmap_g_prev,
                                    distThres_,
                                    angleThres_,
                                    sumDataSE3cp,
                                    outDataSE3cp,
                                    A_icp.data(),
                                    b_icp.data(),
                                    &inlier_check[0],
                                    &corres_map[0],
                                    GPUConfig::getInstance().icpStepThreads,
                                    GPUConfig::getInstance().icpStepBlocks
                                    );
                        int n = sizeof(corres_map)/sizeof(corres_map[0]);
                        int all_num = 0;
                        for(int z = 0; z<n;z++){
                            all_num += corres_map[z];
                        }
                        std::cout<<"Map Check Corres: "<<all_num<<"\n";
                        std::cout<<"Inlier Check Corres: "<<inlier_check[1]<<"\n";
                        //sample const number--------------------------------
                        int samp_point = 50;
                        //int samp_point = (int)all_num*0.2;
                        if (all_num>samp_point){
                            int corres_idx[all_num] = {0};
                            int idx = 0;
                            for(int z = 0; z<n;z++){
                                if(corres_map[z] == 1){
                                    corres_idx[idx] = z;
                                    idx++;
                                }
                            } 
                            int total = 0;
                            int idx_rand = 0;
                            while (total<samp_point)
                            {   
                                int final_idx;
                                idx_rand = rand()%all_num;
                                final_idx = corres_idx[idx_rand];
                                if(corres_map_out[final_idx] == 0){
                                    corres_map_out[final_idx] = 1;
                                    total++;
                                }

                                //std::cout<<"final idx: "<<final_idx<<"\n";
                            }
                            int after_all_num = 0;
                            for(int z = 0; z<n;z++){
                                after_all_num += corres_map_out[z];
                            }
                            std::cout<<"after all num:"<<after_all_num<<"\n";
                            

                        
                            icpStepCorresMap(   
                                        device_Rcurr,
                                        device_tcurr,
                                        vmap_curr,
                                        nmap_curr,
                                        device_Rprev_inv,
                                        device_tprev,
                                        intr(i),
                                        vmap_g_prev,
                                        nmap_g_prev,
                                        distThres_,
                                        angleThres_,
                                        sumDataSE3cp,
                                        outDataSE3cp,
                                        A_icp.data(),
                                        b_icp.data(),
                                        &residual[0],
                                        &corres_map_out[0],
                                        GPUConfig::getInstance().icpStepThreads,
                                        GPUConfig::getInstance().icpStepBlocks
                                        );
                        }
                        else{
                            std::cout<<"Too Less~~ Pass!!!!\n\n";
                            sample_num = 1;
                            samp_flag = false;
                            icpStep(device_Rcurr,
                                device_tcurr,
                                vmap_curr,
                                nmap_curr,
                                device_Rprev_inv,
                                device_tprev,
                                intr(i),
                                vmap_g_prev,
                                nmap_g_prev,
                                distThres_,
                                angleThres_,
                                sumDataSE3,
                                outDataSE3,
                                A_icp.data(),
                                b_icp.data(),
                                &residual[0],
                                GPUConfig::getInstance().icpStepThreads,
                                GPUConfig::getInstance().icpStepBlocks
                                );
                            std::cout<<"too less sample num: "<<residual[1]<<"\n";
                        } 
                    }
                    else{
                         icpStep(device_Rcurr,
                                device_tcurr,
                                vmap_curr,
                                nmap_curr,
                                device_Rprev_inv,
                                device_tprev,
                                intr(i),
                                vmap_g_prev,
                                nmap_g_prev,
                                distThres_,
                                angleThres_,
                                sumDataSE3,
                                outDataSE3,
                                A_icp.data(),
                                b_icp.data(),
                                &residual[0],
                                GPUConfig::getInstance().icpStepThreads,
                                GPUConfig::getInstance().icpStepBlocks
                                );
                    }
                    //std::cout<<"cols: "<<vmap_curr.cols ()<<", rows: "<<vmap_curr.rows ()<<"\n";
                    TOCK("icpStep");
                }

                lastICPError = sqrt(residual[0]) / residual[1];
                lastICPCount = residual[1];
                dA_icp = A_icp.cast<double>();
                db_icp = b_icp.cast<double>();               
                if(icp && rgb)
                {
                    double w = icpWeight;
                    lastA = dA_rgbd + w * w * dA_icp;
                    lastb = db_rgbd + w * db_icp;
                    result = lastA.ldlt().solve(lastb);
                }
                else if(icp)
                {
                    lastA = dA_icp;
                    lastb = db_icp;
                    result = lastA.ldlt().solve(lastb);
                }
                else if(rgb)
                {
                    lastA = dA_rgbd;
                    lastb = db_rgbd;
                    result = lastA.ldlt().solve(lastb);
                }
                else
                {
                    assert(false && "Control shouldn't reach here");
                }

                result_all[k] = result;
                Eigen::Isometry3f rgbOdom;

                OdometryProvider::computeUpdateSE3Fake(resultRt, result, rgbOdom);

                Eigen::Isometry3f currentT;
                currentT.setIdentity();
                currentT.rotate(Rprev);
                currentT.translation() = tprev;

                currentT = currentT * rgbOdom.inverse();
            
                tcurr = currentT.translation();
                Rcurr = currentT.rotation();

                //tcurr = tcurr_inc + tcurr;
                //Rcurr = Rcurr_inc * Rcurr;
                //std::cout<<"task "<<k<<" : \n"<< Rcurr <<std::endl;
                euler_angles[k] = Rcurr.eulerAngles(2,1,0);
                trans[k] = tcurr;

                std::cout<<"task eular "<<k<<" : "<< euler_angles[k].transpose() <<std::endl;

                mat33 device_Rcurr_tmp = Rcurr;
                float3 device_tcurr_tmp = *reinterpret_cast<float3*>(tcurr.data());
                float inlier2[2] = {0,0};
                char corres_flag[vmap_curr.cols()*vmap_curr.rows()/3] = {0};
                if(samp_flag){
                    GetCorresStep(  device_Rcurr,
                                    device_tcurr,
                                    device_Rcurr_tmp,
                                    device_tcurr_tmp,
                                    vmap_curr,
                                    nmap_curr,
                                    device_Rprev_inv,
                                    device_tprev,
                                    intr(i),
                                    vmap_g_prev,
                                    nmap_g_prev,
                                    distThres_,
                                    angleThres_,
                                    sumDataSE3cp,
                                    outDataSE3cp,
                                    A_icp.data(),
                                    b_icp.data(),
                                    &inlier2[0],
                                    &corres_flag[0],
                                    GPUConfig::getInstance().icpStepThreads,
                                    GPUConfig::getInstance().icpStepBlocks
                                    );
                    int n = sizeof(corres_flag)/sizeof(corres_flag[0]);
                    int all_num = 0;
                    for(int z = 0; z<n;z++){
                        all_num += corres_flag[z];
                    }
                    std::cout<<"all corres num: "<<all_num<<"\n";
                    //std::cout<<"all sampled count: "<<sampled_count<<"\n";           
                    //float inlier_ratio = inlier2[1]/(vmap_curr.cols()*vmap_curr.rows()/3);
                    float inlier_ratio = all_num;
                    std::cout<<"inlier :"<<inlier2[1]<<"\n";
                    std::cout<<"inlier ratio:"<<all_num/inlier2[1]<<"\n";
                    inlier_ratio_all[k] = inlier_ratio;
                    //sample number calculate
                    if(samp_flag){ 
                        int sample_num_update = sample_num;
                        sample_num_update = std::log10(1-0.95)/std::log10(1-std::pow(all_num/inlier2[1], 25));
                        if(sample_num_update<sample_num && sample_num_update>0){
                            std::cout<<"sample num update:"<<sample_num_update<<"\n";
                            //sample_num = sample_num_update;
                        }
                        /*
                        else if(sample_num_update>sample_num){
                            std::cout<<"sample num update:"<<sample_num_update<<"\n";
                            if(sample_num_update<850){
                                sample_num = sample_num_update;
                            }
                            else{
                                sample_num = 1200;
                            }
                        }
                        */
                    }

                }
                /*
                for(int af = 0;af<120*160;af++){
                    std::cout<<corres_flag[af];
                }
                */
                //std::cout<<"\n";
               //std::cout<<"task back : \n"<< m <<std::endl;

            }
            
            //std::cout<<"task eular "<<0<<" : "<< euler_angles[0].transpose() <<std::endl;
            Eigen::Vector3f rotAll(0,0,0);
            Eigen::Vector3f transAll(0,0,0);
            //Eigen::Vector3f rotTmp(0,0,0);
            //Eigen::Vector3f transTmp(0,0,0);

            //float inlier_ratio_tmp;
            //get rid of outlier pose
            double rot_score [sample_num];
            double rot_score_sum;
            double trans_score [sample_num];
            double trans_score_sum;
            int final_num = sample_num;

            for ( int a = sample_num - 1; a>0; a--){
                for(int s = 0; s < a - 1; s++){
                    if (inlier_ratio_all[s]<inlier_ratio_all[s + 1]){
                        Eigen::Vector3f rotTmp = euler_angles[s];
                        euler_angles[s] = euler_angles[s + 1];
                        euler_angles[s + 1] = rotTmp;

                        Eigen::Vector3f transTmp = trans[s];
                        trans[s] = trans[s + 1];
                        trans[s + 1] = transTmp;

                        float inlier_ratio_tmp = inlier_ratio_all[s];
                        inlier_ratio_all[s] = inlier_ratio_all[s + 1];
                        inlier_ratio_all[s + 1] = inlier_ratio_tmp;

                        Eigen::Matrix<double, 6, 1> result_tmp = result_all[s];
                        result_all[s] = result_all[s + 1];
                        result_all[s + 1] = result_all[s];

                    }
                }

            }
            //get rid of outlier pose end
            rotAll = euler_angles[0];
            transAll = trans[0];
            //result = result_all[0];
            /*
            if (i == 0){
                rotAll = euler_angles[0];
                transAll = trans[0];
            }
            else{
                rotAll = (euler_angles[0] + euler_angles[1] + euler_angles[2])/3;
                transAll = (trans[0] + trans[1] + trans[2])/3;
            }
            */
            //std::cout<<"task before eular all : "<< rotAll.transpose()<<std::endl;
            //rotAll = euler_angles[9];
            //transAll = trans[9];
            std::cout<<"task eular all : "<< rotAll.transpose()<<std::endl;
            std::cout<<"max inlier ratio:"<< inlier_ratio_all[0]<<"\n";
            Eigen::Matrix<float, 3, 3, 1> resultA;

            resultA = Eigen::AngleAxisf(rotAll.transpose()[0], Eigen::Vector3f::UnitZ())
            * Eigen::AngleAxisf(rotAll.transpose()[1], Eigen::Vector3f::UnitY())
            * Eigen::AngleAxisf(rotAll.transpose()[2], Eigen::Vector3f::UnitX());
            //std::cout<<"task back : "<< resultA <<std::endl;

            //tcurr = transAll;
            //Rcurr = resultA;
            if(samp_flag){

                Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_icp;
                Eigen::Matrix<float, 6, 1> b_icp;;
                

                mat33 device_Rcurr_tmp = resultA;
                char corres_flag[vmap_curr.cols()*vmap_curr.rows()/3] = {0};

                float3 device_tcurr_tmp = *reinterpret_cast<float3*>(transAll.data());
                
                float residual[2] = {0,0};
                GetCorresStep(          device_Rcurr,
                                        device_tcurr,
                                        device_Rcurr_tmp,
                                        device_tcurr_tmp,
                                        vmap_curr,
                                        nmap_curr,
                                        device_Rprev_inv,
                                        device_tprev,
                                        intr(i),
                                        vmap_g_prev,
                                        nmap_g_prev,
                                        distThres_,
                                        angleThres_,
                                        sumDataSE3cp,
                                        outDataSE3cp,
                                        A_icp.data(),
                                        b_icp.data(),
                                        &residual[0],
                                        &corres_flag[0],
                                        GPUConfig::getInstance().icpStepThreads,
                                        GPUConfig::getInstance().icpStepBlocks
                                        );
                int n = sizeof(corres_flag)/sizeof(corres_flag[0]);
                int all_num = 0;
                for(int z = 0; z<n;z++){
                    all_num += corres_flag[z];
                }
                std::cout<<"all corres num: "<<all_num<<"\n";
                 
                icpStepCorresMap(   device_Rcurr,
                                    device_tcurr,
                                    vmap_curr,
                                    nmap_curr,
                                    device_Rprev_inv,
                                    device_tprev,
                                    intr(i),
                                    vmap_g_prev,
                                    nmap_g_prev,
                                    distThres_,
                                    angleThres_,
                                    sumDataSE3,
                                    outDataSE3,
                                    A_icp.data(),
                                    b_icp.data(),
                                    &residual[0],
                                    &corres_flag[0],
                                    GPUConfig::getInstance().icpStepThreads,
                                    GPUConfig::getInstance().icpStepBlocks
                                    );

                Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_icp = A_icp.cast<double>();
                Eigen::Matrix<double, 6, 1> db_icp = b_icp.cast<double>();
                lastICPError = sqrt(residual[0]) / residual[1];
                lastICPCount = residual[1];
                std::cout<<"inlier count!!!!: "<<lastICPCount<<std::endl;
                if(icp && rgb)
                {
                    double w = icpWeight;
                    lastA = dA_rgbd + w * w * dA_icp;
                    lastb = db_rgbd + w * db_icp;
                    result = lastA.ldlt().solve(lastb);
                }
                else if(icp)
                {
                    lastA = dA_icp;
                    lastb = db_icp;
                    result = lastA.ldlt().solve(lastb);
                }
                else if(rgb)
                {
                    lastA = dA_rgbd;
                    lastb = db_rgbd;
                    result = lastA.ldlt().solve(lastb);
                }
                else
                {
                    assert(false && "Control shouldn't reach here");
                }

                Eigen::Isometry3f rgbOdom;

                OdometryProvider::computeUpdateSE3Real(resultRt, result, rgbOdom);

                Eigen::Isometry3f currentT;
                currentT.setIdentity();
                currentT.rotate(Rprev);
                currentT.translation() = tprev;

                currentT = currentT * rgbOdom.inverse();

                tcurr = currentT.translation();
                Rcurr = currentT.rotation();
            }
            else{
                std::cout<<"no samp pose:"<<tcurr<<"\n";
                OdometryProvider::computeUpdateSE3(resultRt, result);
            }
        }
        //iteration loop
    }//pyrimid loop

    if(rgb && (tcurr - tprev).norm() > 0.3)
    {
        Rcurr = Rprev;
        tcurr = tprev;
    }

    if(so3)
    {
        for(int i = 0; i < NUM_PYRS; i++)
        {
            std::swap(lastNextImage[i], nextImage[i]);
        }
    }

    trans = tcurr;
    rot = Rcurr;
}

Eigen::MatrixXd RGBDOdometry::getCovariance()
{
    return lastA.cast<double>().lu().inverse();
}
