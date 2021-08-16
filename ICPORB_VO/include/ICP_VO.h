/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.


* File Name : ICP_VO.h

* Purpose :

* Creation Date : 2020-08-19

* Last Modified : 廿廿年九月一日 (週二) 廿一時廿八分八秒

* Created By : Ji-Ying, Li 

*_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.*/
#ifndef ICP_VO_H
#define ICP_VO_H 
#include <Eigen/Core>
#include <Eigen/LU> // Matrix inverse()
#include "EAS_ICP.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

class ICP_VO
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar=double;
  using Cloud = Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor>;
  using KeyCloud = Eigen::Matrix<Scalar, Eigen::Dynamic, 6, Eigen::RowMajor>;
  using CurrentCloud = Cloud;
  using Pose = Eigen::Matrix<Scalar, 4, 4, Eigen::RowMajor>;
  using RelativePose = Eigen::Matrix<Scalar, 4, 4, Eigen::RowMajor>;
  using Poses = std::vector<Pose, Eigen::aligned_allocator<Pose>>;
  struct KeyFrame{
    KeyFrame(int _id,
             KeyCloud _keyCloud,
             double _timestamp,
             Pose _pose):
      id(_id),
      keyCloud(_keyCloud),
      timestamp(_timestamp),
      pose(_pose)
    {}
    int id;
    KeyCloud keyCloud;
    double timestamp;
    Pose pose;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
public:
  /*! Constructor of ICP VO 
   *
   *  Initialize the parameters.
   *
   *  @param values: Setting file path
   */
  explicit ICP_VO (const std::string& strSettings);
  virtual ~ICP_VO ();

  /*! Track new frame
   *
   *  Tracking camera pose by depth image
   *
   *  @param values: depth image, timestamp
   *  @return: Current camera pose
   */
  const Pose& Track(const cv::Mat& depth, const double& timestamp);

  /*! Track new frame
   *
   *  Tracking camera pose by depth image
   *
   *  @param values: depth image, cloud, timestamp
   *  @return: Current camera pose
   */
  const Pose& Track(const Cloud& tgtCloud, const double& timestamp);

  /*! Get poses
   *
   *  Obtain a vector of all previous poses.
   *
   *  @return: vector of poses.
   */
  const Poses& GetPoses(){
    return mPoses;
  }

  bool IsValid() {
    return pICP->isValid();
  }

private:
  //methods
  bool CheckUpdateKeyFrame(const RelativePose&);
  CurrentCloud ComputeCurrentCloud(const cv::Mat& depth);
  Eigen::Vector<Scalar, 6> TranslationAndEulerAnglesFromSe3(const RelativePose&);
  const Pose& Track(const double& timestamp);

  //datas
  Poses mPoses;
  RelativePose rpK2C;
  std::unique_ptr<KeyFrame> pKeyFrame;
  bool bPredition, bUseBackup;
  Scalar mThresKeyframeUpdateRot;
  Scalar mThresKeyframeUpdateTrans;
  Scalar mDepthMapFactor;
  std::unique_ptr<EAS_ICP> pICP;
  std::unique_ptr<CurrentCloud> pCurrentCloud;
  std::pair<double, std::unique_ptr<Cloud>> backupCloud;
};
#endif /* ICP_VO_H */
