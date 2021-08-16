/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.


* File Name : EAS_ICP.h

* Purpose :

* Creation Date : 2020-08-19

* Last Modified : 廿廿年九月一日 (週二) 廿一時廿八分十四秒

* Created By : Ji-Ying, Li 

*_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.*/
#ifndef EAS_ICP_H
#define EAS_ICP_H 
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <random>

class EAS_ICP
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar=double;
  using Cloud= Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor>;
  using SourceCloud = Eigen::Matrix<Scalar, Eigen::Dynamic, 6, Eigen::RowMajor>;
  using TargetCloud = Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor>;
  using Transform = Eigen::Matrix<Scalar, 4, 4, Eigen::RowMajor>;
  /*! Constructor of EAS_ICP 
   *
   *  Initialize the parameters.
   *
   *  @param values: Setting file path
   */
  explicit EAS_ICP (const std::string&);
  ~EAS_ICP ();

  /*! Edge-aware sampling
   *
   *  The details in JY's thesis.
   *
   *  @param values: Ordered point cloud
   *  @return: sampled cloud
   */
  const SourceCloud& EdgeAwareSampling(const Cloud& );


  /*! Calculate transformation between two cloud.
   *
   *  ICP iterative procedure.
   *
   *  @param values: source cloud, ordered target cloud, initial guess of transformation
   *  @return: transformation
   */
  const Transform& Register(const SourceCloud& srcCloud, const TargetCloud& tgtCloud, const Transform& initialGuess);

  /*! Check ICP valid or not
   *
   *  It depends on sliding extent, detailed in JY's thesis
   *
   *  @return: true if valid; false, otherwise.
   */
  
  bool isValid(){
    return valid;
  }

  int width, height, pixelSize;
  double fx, fy, cx, cy;
private:

  //basic methods
  Transform ConstructSE3(const Eigen::Vector<Scalar, 6> rt6D);
  bool CheckConverged(const Eigen::Vector<Scalar, 6>& rt6D);

  //sampling methods
  void EdgeDetection(const Cloud& cloud, cv::Mat& edge_mat, cv::Mat& nan_mat, cv::Mat& rej_mat);
  void GeometryWeightingFunction(const cv::Mat& x, cv::Mat& out);
  void PointRejectionByDepthRangeAndGeometryWeight( const Cloud& cloud, const cv::Mat& edgeDistanceMat, const cv::Mat& rej_mat, std::vector<int>& , std::vector<double>& weights);
  void WeightedRandomSampling(int sampling_size, const std::vector<int>& srcInds, const std::vector<double>& weights, std::vector<int>& out);
  const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> KinectNoiseWighting(const SourceCloud& cloud_) {
    int size = cloud_.rows();
    auto depth = Eigen::Map< const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> , Eigen::Unaligned, Eigen::Stride<1,6>> (cloud_.data()+2, size) ;
    return 1.0 / (0.0012 + (depth.array()-0.4).pow(2)*0.0019);
  }
  void CalculateNormal(const Cloud& cloud, const std::vector<int>& sampling_inds, SourceCloud& srcCloud);

  //Matching methods
  bool MatchingByProject2DAndWalk(const SourceCloud& srcCloud, const TargetCloud& tgtCloud);

  Eigen::Vector<EAS_ICP::Scalar, 6> MinimizingP2PLErrorMetric(const SourceCloud& srcCloud, const TargetCloud& tgtCloud, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& weights);
  //basic datas
  Transform rtSE3;
  bool valid;
  double icp_converged_threshold_rot;
  double icp_converged_threshold_trans;
  int max_iters;
  int stride, sampling_size;
  double max_depth, min_depth;

  //sampling datas
  int random_seed; //for robust tum evaluation
  double edge_threshold;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> kinectNoiseWeights;
  SourceCloud mSrcCloud;
  Transform finalTransformation;

  //matching datas
  int search_range, search_step;
  int right_bound, left_bound, bottom_bound, top_bound;
  double fixed_threshold_rejection, dynamic_threshold_rejection;
  Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> corrs;

  //Minimizing datas
  double accSlidingExtent;
  double thresAccSlidingExtent;
  double thresEvalRatio;


};
#endif /* EAS_ICP_H */
