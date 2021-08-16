/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.


* File Name : EAS_ICP.cpp

* Purpose :

* Creation Date : 2020-08-19

* Last Modified : Sun Aug 23 17:09:02 2020

* Created By : Ji-Ying, Li 

*_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.*/

#include <EAS_ICP.h>
#include "pcl_normal.hpp"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

EAS_ICP::EAS_ICP( const std::string& strSettings){
  cv::FileStorage FsSettings(strSettings.c_str(), cv::FileStorage::READ);

  //TODO initial with fsettings
  //basic
  width  = FsSettings["Camera.width"];
  height = FsSettings["Camera.height"];
  pixelSize=width*height;
  fx = FsSettings["Camera.fx"];
  fy = FsSettings["Camera.fy"];
  cx = FsSettings["Camera.cx"];
  cy = FsSettings["Camera.cy"];
  max_depth = FsSettings["Sampling.max_depth"];
  min_depth = FsSettings["Sampling.min_depth"];
  icp_converged_threshold_rot   = FsSettings["ICP.icp_converged_threshold_rot"]; 
  icp_converged_threshold_trans = FsSettings["ICP.icp_converged_threshold_trans"]; 
  max_iters = FsSettings["ICP.max_iters"];
  thresAccSlidingExtent= FsSettings["ICP.thresAccSlidingExtent"]; 

  //sampling
  random_seed    = FsSettings["Sampling.random_seed"];
  if (random_seed == -1) {
    random_seed = 0;
  }
  stride         = FsSettings["Sampling.stride"]; 
  edge_threshold = FsSettings["Sampling.edge_threshold"];
  sampling_size  = FsSettings["Sampling.number_of_sampling"];

  //matching
  search_step                 = FsSettings["DataAssociating.search_step"]; 
  search_range                = FsSettings["DataAssociating.search_range"]; 
  dynamic_threshold_rejection = FsSettings["DataAssociating.dynamic_threshold_rejection"]; 
  fixed_threshold_rejection   = FsSettings["DataAssociating.fixed_threshold_rejection"]; 
  top_bound                   = FsSettings["DataAssociating.top_bound"]; 
  bottom_bound                = FsSettings["DataAssociating.bottom_bound"]; 
  left_bound                  = FsSettings["DataAssociating.left_bound"]; 
  right_bound                 = FsSettings["DataAssociating.right_bound"]; 
                                            
  //Minimizing
  thresEvalRatio              = FsSettings["TransformSolver.thresEvalRatio"]; 
}

EAS_ICP::~EAS_ICP(){

}

const EAS_ICP::Transform& EAS_ICP::Register(const SourceCloud& srcCloud, const TargetCloud& tgtCloud, const Transform& initialGuess) {

  //initial parameters
  rtSE3 = initialGuess;
  int iterations = 0;
  accSlidingExtent = 0;
  while (true) {
    iterations+=1;

    //transform source cloud by inital guess
    SourceCloud transformedCloud(srcCloud.rows(), 6) ;
    transformedCloud.leftCols(3) = ((rtSE3.topLeftCorner<3,3>()* srcCloud.leftCols<3>().transpose()).colwise() + rtSE3.col(3).head<3>()).transpose();
    transformedCloud.rightCols(3) = (rtSE3.topLeftCorner<3,3>()* srcCloud.rightCols<3>().transpose()).transpose();

    //match correspondence
    if (!MatchingByProject2DAndWalk(transformedCloud, tgtCloud)) {
      break; // when correspondence size less than 6
    }
    
    //get iteration transformation by minimizing p2pl error metric
    Eigen::Vector<Scalar, 6> rt6D;
    rt6D = MinimizingP2PLErrorMetric(transformedCloud(corrs.col(0), Eigen::all), tgtCloud(corrs.col(1), Eigen::all), kinectNoiseWeights(corrs.col(0), Eigen::all));
    
    //convert 6D vector to SE3
    Transform iterRtSE3;
    iterRtSE3 = ConstructSE3(rt6D);

    //chain iterRtSE3 to rtSE3
    rtSE3 = iterRtSE3 * rtSE3;

    //check termination
    if (CheckConverged(rt6D) || iterations > max_iters) {
      break;
    }
  }
  //justify valid by sliding extent
  if (accSlidingExtent < thresAccSlidingExtent) {
    valid = true;
  } else {
    valid = false;
  }
  return rtSE3;
}
void EAS_ICP::EdgeDetection(const Cloud& cloud, cv::Mat& edge_mat, cv::Mat& nan_mat, cv::Mat& rej_mat) {
  Eigen::MatrixX<Scalar> depth_map = Eigen::Map< const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> , Eigen::Unaligned, Eigen::Stride<1,3>> (cloud.data()+2, cloud.rows());
  edge_mat= cv::Mat::zeros(height, width, CV_8UC1);
  nan_mat = cv::Mat::zeros(height, width, CV_8UC1);
  rej_mat = cv::Mat::zeros(height, width, CV_8UC1);
  uchar* edge_map = edge_mat.data,
           *nan_map = nan_mat.data,
           *rej_map = rej_mat.data;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
	    double point_z = depth_map(i*width+j, 0);
      if (point_z != point_z) {
        nan_map[i*width + j] = 255;
      }
	    if (point_z >= max_depth  || point_z <= min_depth ) {
        rej_map[i*width + j] = 255;
	    }
    }
  }
  for (int i = 0; i < height; i+=stride) {
    int last_pixel_index = -1;
    double last_pixel_z = -1;
    for (int j = 0; j < width; j+=stride) {
	    double point_z = depth_map(i*width+j, 0);
      if (point_z != point_z) {
        continue;
      }
	    if (point_z >= max_depth  || point_z <= min_depth ) {
          continue;
	    }
      if (last_pixel_index >= 0) {
        double pixel_min = std::min(last_pixel_z, point_z);
        double threshold = edge_threshold* pixel_min * (abs(j - last_pixel_index));
        if (fabs(point_z - last_pixel_z) > threshold)
        {
          edge_map[i*width + j] = 255;
          edge_map[i*width + last_pixel_index] = 255;
        }
      }
      last_pixel_index = j;
      last_pixel_z = point_z;
    }
  }
  for (int j = 0; j < width; j+=stride) {
    int last_pixel_index = -1;
    double last_pixel_z = -1;
    for (int i = 0; i < height; i+=stride) {
	    double point_z = depth_map(i*width+j, 0);
      if (point_z != point_z) {
        continue;
      }
	    if (point_z >= max_depth  || point_z <= min_depth ) {
          continue;
	    }
      if (last_pixel_index >= 0) {
        double pixel_min = std::min(last_pixel_z, point_z);
        double threshold = edge_threshold* pixel_min * (abs(i - last_pixel_index));
        if (fabs(point_z - last_pixel_z) > threshold)
        {
          edge_map[i*width + j] = 255;
          edge_map[last_pixel_index*width + j] = 255;
        }
      }
      last_pixel_index = i;
      last_pixel_z = point_z;
    }
  }
  
}
void EAS_ICP::GeometryWeightingFunction(const cv::Mat& x, cv::Mat& out) {
  std::vector<double> geometry_weighting_coeff{4.55439848e-03, 7.06670913e+00, 1.08456189e+00};
  cv::Mat powmat;
  cv::pow(geometry_weighting_coeff[1] - x, 2.0, powmat);
  out = 1/(geometry_weighting_coeff[0] * powmat + geometry_weighting_coeff[2]);
}

void EAS_ICP::PointRejectionByDepthRangeAndGeometryWeight( const Cloud& cloud, const cv::Mat& edgeDistanceMat, const cv::Mat& rej_mat, std::vector<int>& remindPointIndexes, std::vector<double>& weights){
  cv::Mat geo_weight;
  GeometryWeightingFunction(edgeDistanceMat, geo_weight);
  float* fw_ptr = (float*)geo_weight.data;
  uchar* rej_ptr = (uchar*)rej_mat.data;
  int size = edgeDistanceMat.rows*edgeDistanceMat.cols*edgeDistanceMat.channels();
  remindPointIndexes.resize(size);
  weights.resize(size);
  int cnt = 0;
  for (int i = 0; i < size; ++i) {
    if (rej_ptr[i] == 0) {
        remindPointIndexes[cnt] = i;
        weights[cnt] = (fw_ptr[i]);
        ++cnt;
    }
  }
  weights.resize(cnt);
  remindPointIndexes.resize(cnt);
}
void EAS_ICP::WeightedRandomSampling(int sampling_size, const std::vector<int>& roiInds, const std::vector<double>& weights, std::vector<int>& samplingInds) {
  size_t N = roiInds.size ();  
  if (sampling_size >= N)
  {
    samplingInds = roiInds;
  } else {
    samplingInds.resize (sampling_size);
    
    std::default_random_engine gen; 
    gen.seed(random_seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Algorithm S
    size_t i = 0;
    size_t index = 0;
    std::vector<bool> added;
    size_t n = sampling_size;
    double sum = Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned>(weights.data(), N).sum();
    while (n > 0)
    {
      // Step 1: [Generate U.] Generate a random variate U that is uniformly distributed between 0 and 1.
      //const float U = rand()/ double(RAND_MAX);
      const float U = dis(gen);
      double prob = weights[index] /sum;
      // std::csamplingInds << norm_test << std::endl;
      // Step 2: [Test.] If N * U > n, go to Step 4. 
      //if (N <= n) {
      //  samplingInds[i++] = roiInds[index];
      //  --n;
      //} else
       if ((U) <= n*(prob)) {
        samplingInds[i++] = roiInds[index];
        --n;
      }
      --N;
      sum = sum - weights[index];
      ++index;
    }
  }
}

void EAS_ICP::CalculateNormal( const Cloud& cloud,  const std::vector<int>& samplingInds, SourceCloud& srcCloud){

  const int normal_step = 2;
  const int normal_range = 3;
  int size = samplingInds.size();
  int low_bound= - normal_range;
  int up_bound= normal_range + 1;
  std::vector<int> validNormalInds;
  validNormalInds.reserve(size);
  Cloud normals(size, 3);
  typedef PointXYZ PointT;
  for (int i = 0; i < size; ++i) {
    int x = samplingInds[i] % width;
    int y = samplingInds[i] / width;
    std::vector<PointT> points;

    //reserve range size memory
    points.reserve(std::pow(2*normal_range+1, 2));

    //center epoint
    PointT p (
            cloud(samplingInds[i], 0),
            cloud(samplingInds[i], 1),
            cloud(samplingInds[i], 2)
        );
    for (int ix = low_bound; ix < up_bound; ++ix){
      for (int iy = low_bound; iy < up_bound; ++iy)
      {
        int x_ = x + ix*normal_step;
        int y_ = y + iy*normal_step;
        if ( (x_ >= width)
          || (y_ >= height)
          || (x_ < 0)
          || (y_ < 0) )
        {
          continue;
        }

        // use non-sampled point cloud points
        const PointT  original_p(
            cloud(y_ * width + x_, 0),
            cloud(y_ * width + x_, 1),
            cloud(y_ * width + x_, 2)
            );

        //check nan
        if(original_p.z != original_p.z){
          continue;
        }

        // skip the larger normal (cloud be noise)
        if (fabs(original_p.z - p.z) > normal_step*3*p.z/ fx )
        {
          continue;
        }
        // pick the points
        points.push_back(original_p);
      }
    }
    // compute the normal of ref points
    Eigen::Vector4f plane_param_out;
    float curvature_out;
    computePointNormal(points, plane_param_out, curvature_out);

    // correct the normal, set the view point at (0,0,0)
    flipNormalTowardsViewpoint<PointT, float>(p, 0, 0, 0, plane_param_out);
    normals(i, 0) = plane_param_out[0];
    normals(i, 1) = plane_param_out[1];
    normals(i, 2) = plane_param_out[2];
    if (!plane_param_out.hasNaN()) {
      validNormalInds.push_back(i);
    }
  }

  //construct Source Cloud
  std::vector<int> validSrcCloudInds(validNormalInds.size());
  for (int i = 0; i < validNormalInds.size(); ++i) {
    validSrcCloudInds[i] = samplingInds[validNormalInds[i]];
  }

  SourceCloud tmp(validSrcCloudInds.size(), 6); 
  tmp.leftCols(3) = cloud(validSrcCloudInds, Eigen::all);
  tmp.rightCols(3) = normals(validNormalInds, Eigen::all);
  tmp.swap(srcCloud);
}
const EAS_ICP::SourceCloud& EAS_ICP::EdgeAwareSampling(const Cloud& cloud) {
  //sampling
  //edge detection
  cv::Mat edge_map;
  cv::Mat nan_map;
  cv::Mat rej_map;
  EdgeDetection(cloud, edge_map, nan_map, rej_map);
  
  //calculate edge distance map
  cv::Mat inv_edge_map;
  inv_edge_map = ~edge_map;
  cv::Mat edge_distance_map;
  cv::distanceTransform(inv_edge_map, edge_distance_map, cv::DIST_L2, 5);
  
  //reject the points out of range and weight remind points for random sampling
  std::vector<int> remindPointInds;
  std::vector<double> weights;
  PointRejectionByDepthRangeAndGeometryWeight(cloud, edge_distance_map, nan_map | rej_map, remindPointInds, weights);

  //sample depend on edge distance
  std::vector<int> EASInds;
  WeightedRandomSampling(sampling_size, remindPointInds, weights, EASInds);
  
  //calculate normal
  CalculateNormal(cloud, EASInds, mSrcCloud);
  
  //weighting
  kinectNoiseWeights = KinectNoiseWighting(mSrcCloud);
  
  return mSrcCloud;
}

bool EAS_ICP::MatchingByProject2DAndWalk(const SourceCloud& srcCloud, const TargetCloud& tgtCloud) {
  int size = srcCloud.rows();
  //correspondence declare
  std::vector<std::tuple<Scalar, int, int>> stdCorrs; 
  std::vector<double> residual_vector;

  int corr_cnt= 0;

  //for all correspondence
  for (int i = 0; i < size; ++i) {
    //a source point
    const Scalar& src_px = srcCloud(i, 0);
    const Scalar& src_py = srcCloud(i, 1);
    const Scalar& src_pz = srcCloud(i, 2);

    // project to 2D target frame
    int x_warp = fx / src_pz*src_px +cx;
    int y_warp = fy / src_pz*src_py +cy;
    //declare and initial variables
    Scalar min_distance = std::numeric_limits<Scalar>::max();
    int target_index = -1;

    //check the 2D point in target frame range
    if (x_warp >= width || y_warp >= height || x_warp < 0 || y_warp < 0)
    {
      continue;
    }

    //search range
    for (int ix = -search_range; ix < search_range + 1 ; ++ix)
    {
      for (int iy = -search_range; iy < search_range + 1 ; ++iy)
      {
        // search a circle range
        int grid_distance2 = ix*ix + iy*iy;
        if (grid_distance2 > search_range* search_range)
        {
          continue;
        }
        // x index and y index of target frame
        int x_idx = x_warp + ix * search_step;
        int y_idx = y_warp + iy * search_step;

        // avoid index out of target frame
        if (x_idx >= (width)
          || x_idx < 0
          || y_idx >=height
          || y_idx < 0)
        {
          continue;
        }

        //calculate 1D target frame index
        int tmp_index = (y_idx * width + x_idx);

        // get x,y,z of target point
        double tgt_px = tgtCloud(tmp_index,0);
        double tgt_py = tgtCloud(tmp_index,1);
        double tgt_pz = tgtCloud(tmp_index,2);

        //check nan
        if(
             (tgt_px!= tgt_px)||
             (tgt_py!= tgt_py)||
             (tgt_pz!= tgt_pz)
          ) continue; 


        //calculate the distance between source point and target point
        double distance = sqrt((src_px - tgt_px)*(src_px - tgt_px)
          + (src_py - tgt_py)*(src_py - tgt_py)
          + (src_pz - tgt_pz)*(src_pz - tgt_pz));

        // if new distance is less than min distance => record this index and distance
        if (distance < min_distance)
        {
          min_distance = distance;
          target_index = tmp_index;// target index: height x width x pointsize //pointsize is 6
        }
      }
    }
    //image boundary rejection
    //check closet point whether in the margin of boundary
    int target_x = target_index % width;
    int target_y = target_index / width;
    if (target_x > right_bound|| target_x < left_bound || target_y > bottom_bound || target_y < top_bound) {
      continue;
    }

    //check closet point existed and smaller fix threshold of rejection ==> if true, this pair is correspondence, and store in vector stdCorrs
    if ( min_distance !=  std::numeric_limits<Scalar>::max() && min_distance < fixed_threshold_rejection)
    {
      stdCorrs.push_back(std::make_tuple(min_distance, i, target_index));
      residual_vector.push_back(min_distance);
      ++corr_cnt;
    }
  }
  //dynamic rejction
  //calculate the real index from the ratio of dynamic threshold
  int dynamic_threshold_index = dynamic_threshold_rejection * residual_vector.size();

  //check the index is not over the vector size
  if (dynamic_threshold_index < residual_vector.size()) {
      //rejection theshold(unit:m)
      //calculate the real value corresponded the index
      std::nth_element(residual_vector.begin(), residual_vector.begin() + dynamic_threshold_index, residual_vector.end());
      float reject_distance_threshold = 0; 
      //check the vector is no empty, or segmentation fault would occur.
      if (residual_vector.size() > 0)
        reject_distance_threshold = residual_vector[dynamic_threshold_index];
      
      //erase the correspondence over the dynamic threshold
      stdCorrs.erase(std::remove_if(stdCorrs.begin(), stdCorrs.end(), [reject_distance_threshold](const std::tuple<double, int,int>& elem){ return std::get<0>(elem)>reject_distance_threshold;}), stdCorrs.end());
  }

  //change the type for meeting the function output requirement
  int final_size = stdCorrs.size();
  Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> correspondences(final_size, 2);
  for (int i = 0; i < stdCorrs.size(); ++i) {
    correspondences(i, 0) = std::get<1>(stdCorrs[i]);
    correspondences(i, 1) = std::get<2>(stdCorrs[i]);
  }
  correspondences.swap(corrs);

  //check the correspondence size over 6 for solving 6 rt valuables
  return corrs.rows() >=6;
}


Eigen::Vector<EAS_ICP::Scalar, 6> EAS_ICP::MinimizingP2PLErrorMetric(const SourceCloud& srcCloud, const TargetCloud& tgtCloud, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& weights) {
  //solve Ax=b
  //calculate b
  auto b = (tgtCloud - srcCloud.leftCols(3)).transpose().cwiseProduct(srcCloud.rightCols(3).transpose()).colwise().sum().transpose();
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b_weighted = b.array() * weights.array();

  //calculate A
  Eigen::Matrix<Scalar, Eigen::Dynamic, 6, Eigen::RowMajor> A(tgtCloud.rows(), 6);
  for (int i = 0; i < tgtCloud.rows(); ++i) {
    Eigen::Vector3<Scalar> s = srcCloud.row(i).tail(3);
    A.row(i).head(3) = tgtCloud.row(i).cross(s);
  }
  A.rightCols(3) = srcCloud.rightCols(3);
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A_weighted = A.array().colwise() * weights.array();

  //solve x by svd
  auto _bdcSVD = A_weighted.bdcSvd(Eigen::ComputeThinU |Eigen:: ComputeThinV);
  Eigen::Vector<EAS_ICP::Scalar, 6> ret;
  ret = _bdcSVD.solve(b_weighted);

  //Accumulate sliding extent 
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 6, 6> > evalsolver (A_weighted.transpose()*A_weighted);
  auto norm_eval = evalsolver.eigenvalues()/evalsolver.eigenvalues().maxCoeff();
  std::vector<int> inds;
  for (int i = 0; i < norm_eval.size(); ++i) {
    if (norm_eval(i) < 1.0/thresEvalRatio) {
      inds.push_back(i);
    }
  }
  if (inds.size() > 0) {
    auto less_eval_vecs = evalsolver.eigenvectors()(Eigen::all, inds);
    double sliding_dist = (less_eval_vecs.transpose() * ret).norm();
    accSlidingExtent += sliding_dist;
  }
  return ret;
}

//reference linear least-square Optimization of p2pl ICP
EAS_ICP::Transform EAS_ICP::ConstructSE3(const Eigen::Vector<Scalar, 6> rt){
  Transform ret = Transform::Identity();
  const Scalar & alpha = rt(0); const Scalar & beta = rt(1); const Scalar & gamma = rt(2);
  const Scalar & tx = rt(3);    const Scalar & ty = rt(4);   const Scalar & tz = rt(5);
  ret(0,0)= static_cast<Scalar> ( cos (gamma) * cos (beta));
  ret(0,1)= static_cast<Scalar> (-sin (gamma) * cos (alpha) + cos (gamma) * sin (beta) * sin (alpha));
  ret(0,2)= static_cast<Scalar> ( sin (gamma) * sin (alpha) + cos (gamma) * sin (beta) * cos (alpha));
  ret(1,0)= static_cast<Scalar> ( sin (gamma) * cos (beta));
  ret(1,1)= static_cast<Scalar> ( cos (gamma) * cos (alpha) + sin (gamma) * sin (beta) * sin (alpha));
  ret(1,2)= static_cast<Scalar> (-cos (gamma) * sin (alpha) + sin (gamma) * sin (beta) * cos (alpha));
  ret(2,0)= static_cast<Scalar> (-sin (beta));
  ret(2,1)= static_cast<Scalar> ( cos (beta) * sin (alpha));
  ret(2,2) = static_cast<Scalar> ( cos (beta) * cos (alpha));
  ret(0,3)= static_cast<Scalar> (tx);
  ret(1,3)= static_cast<Scalar> (ty);
  ret(2,3) = static_cast<Scalar> (tz);
  return ret;
}
bool EAS_ICP::CheckConverged(const Eigen::Vector<Scalar, 6>& rt6D) {
    return (sqrt(rt6D.head(3).array().pow(2).sum())  < icp_converged_threshold_rot )&&(sqrt(rt6D.tail(3).array().pow(2).sum())  < icp_converged_threshold_trans );
  
}
