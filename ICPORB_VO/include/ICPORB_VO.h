/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.


* File Name : ICPORB_VO.h

* Purpose :

* Creation Date : 2020-08-19

* Last Modified : 廿廿年九月一日 (週二) 廿一時廿八分一秒

* Created By : Ji-Ying, Li 

*_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.*/
#ifndef ICPORB_VO_H
#define ICPORB_VO_H 
#include <System.h>
#include "ICP_VO.h"
#include <future>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/factory.h>
#include <g2o/core/base_vertex.h>
#include <memory>

class ICPORB_VO
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar=double;
  using Pose = Eigen::Transform<Scalar, 3, Eigen::Isometry, Eigen::RowMajor>;
  using Poses = std::vector<Pose, Eigen::aligned_allocator<Pose>>;
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1,-1>> SlamBlockSolver;
  /*! Constructor of ICPORB VO
   *
   *  Initialize the parameters.
   *
   *  @param values: Setting file path
   */
  explicit ICPORB_VO (const std::string& strVOC, const std::string& strSettings);
  /*! Track new frame
   *
   *  Tracking camera pose by fusing ICP and ORB slam 
   *
   *  @param values: rgb image, depth image, timestamp
   */
  void IncrementalTrack(const cv::Mat& rgb, const cv::Mat& depth, const ICP_VO::Cloud& cloud, const double& timestamp);
  /*! Track new frame
   *
   *  Tracking camera pose by fusing ICP and ORB slam 
   *
   *  @param values: rgb image, depth image, timestamp
   */
  void IncrementalTrack(const cv::Mat& rgb, const cv::Mat& depth, const double& timestamp);
  void Track(const cv::Mat& rgb, const cv::Mat& depth, const double& timestamp);
  Poses FusionTwoTrajectory();
  virtual ~ICPORB_VO ();
  /* Get current pose 
   *
   * @param values: flag to select different way pose.
   *  1 for optimized
   *  2 for ORB
   *  otherwise for ICP
   */
  const Pose GetCurrentPose(int flag){
    if (flag == 1) {
      return currentPose;
    } else if (flag ==2 ) {
      return posesORB.back();
    } else {
      return posesICP.back();
    }
  }
  /* Get current relative pose 
   * @param values: flag to select different way pose.
   *  1 for optimized
   *  2 for ORB
   *  otherwise for ICP
   */
  const Pose GetRelativePose(int flag)
  {
    if (posesICP.size() > 1) {
      if (flag == 1) {
        return lastPose.inverse()*currentPose;
      } else if (flag ==2 ) {
        return posesORB.rbegin()[1].inverse(), posesORB.rbegin()[0];
      } else {
        return posesICP.rbegin()[1].inverse(), posesICP.rbegin()[0];
      }
    } else {
      return Pose::Identity();
    }
  }
  bool IsValid() {
    return valid;
  }
private:
  //fusion methods
  void AddICPEdge();
  void AddORBEdge(int);
  void Fuse(int, bool, bool);
  void TUMTrajectoryReader(const std::string& file, Poses& out, std::vector<std::string>& times);

  //ORB SLAM2 tracking system
  ORB_SLAM2::System* pORBVO;
  //EAS ICP tracking system
  ICP_VO* pICPVO;

  Poses posesORB;
  Poses posesICP;
  Poses posesFusion;


  //optimization parameter
  int orbEdgeLength;
  int orbResetFlag;
  bool valid;
  Pose lastPose, currentPose;

  enum StateORBReset { 
    NORESET, RESET
  };
  StateORBReset stateORBReset;
};
#endif /* ICPORB_VO_H */
