/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.


* File Name : ICP_VO.cpp

* Purpose :

* Creation Date : 2020-08-19

* Last Modified : 廿廿年九月一日 (週二) 廿一時九分一秒

* Created By : Ji-Ying, Li 

*_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.*/

#include "ICP_VO.h"

ICP_VO::ICP_VO(const std::string& strSettings):
  pKeyFrame(nullptr),
  pCurrentCloud(nullptr)
{
  backupCloud.second = nullptr;
  cv::FileStorage FsSettings(strSettings.c_str(), cv::FileStorage::READ);
  pICP = std::make_unique<EAS_ICP> (strSettings);
  //initialize parameters
  //TODO initial from settings
  mDepthMapFactor = FsSettings["DepthMapFactor"];
  bUseBackup = (int)(FsSettings["VO.use_backup"]);
  bPredition = (int)(FsSettings["VO.use_prediction"]);
  mThresKeyframeUpdateRot   = FsSettings["VO.keyframe_update_threshold_rot"];
  mThresKeyframeUpdateTrans = FsSettings["VO.keyframe_update_threshold_trans"];
}

ICP_VO::~ICP_VO() {
}
const ICP_VO::Pose& ICP_VO::Track(const cv::Mat& depth, const double& timestamp){
  //calculate 3D points
  pCurrentCloud = std::make_unique<CurrentCloud>(ComputeCurrentCloud(depth));
  return Track(timestamp);
}
const ICP_VO::Pose& ICP_VO::Track(const Cloud& _cloud, const double& timestamp){
  //calculate 3D points
  pCurrentCloud = std::make_unique<CurrentCloud>(_cloud);
  return Track(timestamp);
}
const ICP_VO::Pose& ICP_VO::Track(const double& timestamp)
{
  //initialize VO if first frame, or track current frame
  if (mPoses.size() == 0) {
    rpK2C = RelativePose::Identity();
    pKeyFrame = std::make_unique<KeyFrame>(mPoses.size(), pICP->EdgeAwareSampling(*pCurrentCloud), timestamp, Pose::Identity());
    mPoses.push_back(Pose::Identity());
  } else {
    //calculate prediction
    RelativePose pred, predK2C;

    //guess the initial pose
    if (bPredition && mPoses.size() > 1) {
      pred = mPoses.rbegin()[0].inverse()* mPoses.rbegin()[1];
      predK2C = pred*rpK2C;
    } else {
      pred = RelativePose::Identity();
      predK2C = RelativePose::Identity();
    }

    //perform ICP
    rpK2C = pICP->Register(pKeyFrame->keyCloud, *pCurrentCloud, predK2C);

    //check ICP valid
    if (bUseBackup && !pICP->isValid() && backupCloud.second != nullptr && mPoses.size() != pKeyFrame->id) {
      pKeyFrame = std::make_unique<KeyFrame>(mPoses.size(), pICP->EdgeAwareSampling(*backupCloud.second), backupCloud.first, mPoses.back());
      RelativePose rpB2C = pICP->Register(pKeyFrame->keyCloud, *pCurrentCloud, pred);
      if (pICP->isValid()) {
        rpK2C = rpB2C;
      } else {
        rpK2C = pred;
      }
    } else if(bUseBackup && !pICP->isValid() && backupCloud.second != nullptr){
      rpK2C = pred;
    }

    //calculate current pose from keypose
    Pose currentPose = pKeyFrame->pose * (rpK2C.inverse());
    mPoses.push_back(currentPose);

    //check update keyframe
    if (CheckUpdateKeyFrame(rpK2C)) {
      pKeyFrame = std::make_unique<KeyFrame>(mPoses.size(), pICP->EdgeAwareSampling(*pCurrentCloud), timestamp, mPoses.back());
      rpK2C = RelativePose::Identity();
    }
    
    //backup cloud
    if (bUseBackup) {
      backupCloud.first = timestamp;
      backupCloud.second = std::move(pCurrentCloud);
    } 
  }
  return mPoses.back();
}
bool ICP_VO::CheckUpdateKeyFrame(const RelativePose& rtSE3) {
  Eigen::Vector<Scalar, 6> rt;
  rt = TranslationAndEulerAnglesFromSe3(rtSE3);
  Scalar trans = sqrtf(rt.tail(3).array().pow(2).sum());
  Scalar rot = sqrtf(rt.head(3).array().pow(2).sum());
  if ( (trans > mThresKeyframeUpdateTrans) || (rot > mThresKeyframeUpdateRot)) {
    return true;
  }
  else {
    return false;
  }
}
Eigen::Vector<ICP_VO::Scalar, 6> ICP_VO::TranslationAndEulerAnglesFromSe3(const RelativePose& rtSE3) {
  Eigen::Vector<ICP_VO::Scalar, 6> ret;
  Scalar &x = ret(3), &y = ret(4), &z = ret(5);
  Scalar &roll = ret(0), &pitch = ret(1), &yaw = ret(2);
  x = rtSE3 (0, 3);
  y = rtSE3 (1, 3);
  z = rtSE3 (2, 3);
  roll = std::atan2 (rtSE3 (2, 1), rtSE3 (2, 2));
  pitch = asin (-rtSE3 (2, 0));
  yaw = std::atan2 (rtSE3 (1, 0), rtSE3 (0, 0));
  return ret;
}
ICP_VO::CurrentCloud ICP_VO::ComputeCurrentCloud(const cv::Mat& depth) {
  int width = pICP->width;
  int height = pICP->height;
  CurrentCloud ret(pICP->pixelSize, 3);
  for (int n = 0; n < height; ++n) {
    for (int m = 0; m < width; ++m) {
      int i = n*width +m;
      Scalar z=0;
      if (depth.type() == CV_32FC1) {
         // z = ((float*)depth.data)[i]/ mDepthMapFactor;
         z = mDepthMapFactor;
      } else {
         z = ((int16_t*)depth.data)[i]/ mDepthMapFactor;
      }
      if (z == 0) {
        for (int j = 0; j < 3; ++j) {
          ret(i, j) = std::numeric_limits<Scalar>::quiet_NaN();
        }
      } else {
        ret(i, 2) = z;
        Scalar x = (Scalar)m;
        Scalar y = (Scalar)n;
        ret(i, 0) = z*(x-pICP->cx) / pICP->fx;
        ret(i, 1) = z*(y-pICP->cy) / pICP->fy;
      }
    }
  }
  return ret;
}

