/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.


* File Name : ICPORB_VO.cpp

* Purpose :

* Creation Date : 2020-08-19

* Last Modified : 廿廿年九月一日 (週二) 廿一時十一分三秒

* Created By : Ji-Ying, Li 

*_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.*/

#include <ICPORB_VO.h>

ICPORB_VO::ICPORB_VO (const std::string& strVOC, const std::string& strSettings)
{
  std::cout << "VOC Path: " << strVOC << std::endl;
  std::cout << "Setting File Path: " << strSettings << std::endl;
  cv::FileStorage FsSettings(strSettings.c_str(), cv::FileStorage::READ);
  bool drawORB = (int)FsSettings["Viewer.Enable"];
  orbEdgeLength = (int)FsSettings["VO.orbEdgeLength"];
  orbResetFlag = false;
  stateORBReset = StateORBReset::NORESET;

  pORBVO = new ORB_SLAM2::System(strVOC, strSettings, ORB_SLAM2::System::eSensor::RGBD, drawORB);
  pICPVO = new ICP_VO(strSettings);
}

ICPORB_VO::~ICPORB_VO (){
#define DELETE_PTR_OF_ICPORBVO(_ptr)  \
  if((_ptr)!= nullptr){ \
    delete _ptr; \
    (_ptr) = nullptr; \
  }
  // DELETE_PTR_OF_ICPORBVO(pORBVO);
  DELETE_PTR_OF_ICPORBVO(pICPVO);
}

void ICPORB_VO::Fuse(int n, bool orb_valid, bool icp_valid) {
  // if(!orb_valid && !icp_valid){
  if(!icp_valid){
    valid = false;
    return;
  } else {
    valid = true;
  }
  
  g2o::SparseOptimizer optimizer;
  auto linear_solver = std::make_unique< g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType>>();
  auto block_solver = std::make_unique< SlamBlockSolver>(std::move(linear_solver));
  g2o::OptimizationAlgorithmGaussNewton * solver = new g2o::OptimizationAlgorithmGaussNewton( std::move(block_solver));
  optimizer.setAlgorithm(solver);
  {
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId(1);
    v->setEstimate(currentPose);
    v->setFixed(true);
    optimizer.addVertex(v);
  }
  for (int i = 0; i < n; ++i) {
    if (i != 1) {
      g2o::VertexSE3* v = new g2o::VertexSE3();
      v->setId(i);
      v->setEstimate(posesICP.rbegin()[i]);
      optimizer.addVertex(v);
    }
  }
  //for only one icp edge
  if (icp_valid) {
    g2o::EdgeSE3* e = new g2o::EdgeSE3();
    auto from = optimizer.vertex(0);
    auto to = optimizer.vertex(1);
    e->vertices()[0] = from;
    e->vertices()[1] = to;
    e->setMeasurement(posesICP.rbegin()[0].inverse() * posesICP.rbegin()[1]);
    e->setInformation(Eigen::MatrixX<Scalar>::Identity(6,6));
    optimizer.addEdge(e);
  }
  // for (int i = 1; i < n; ++i) {
  //   g2o::EdgeSE3* e = new g2o::EdgeSE3();
  //   auto from = optimizer.vertex(i-1);
  //   auto to = optimizer.vertex(i);
  //   e->vertices()[0] = from;
  //   e->vertices()[1] = to;
  //   e->setMeasurement(posesICP.rbegin()[i-1].inverse() * posesICP.rbegin()[i]);
  //   e->setInformation(Eigen::MatrixX<Scalar>::Identity(6,6));
  //   optimizer.addEdge(e);
  // }
  // int i = posesORB.size() - 1;
  if (orb_valid) {
    g2o::EdgeSE3* e = new g2o::EdgeSE3();
    auto from = optimizer.vertex(n-1);
    auto to = optimizer.vertex(0);
    e->vertices()[0] = from;
    e->vertices()[1] = to;
    e->setMeasurement(posesORB.rbegin()[n-1].inverse() * posesORB.rbegin()[0]);
    e->setInformation(Eigen::MatrixX<Scalar>::Identity(6,6));
    optimizer.addEdge(e);
  }
  // optimizer.setVerbose(true);
  optimizer.initializeOptimization();
  optimizer.optimize(2);
  lastPose = currentPose;
  currentPose = dynamic_cast<g2o::VertexSE3*> (optimizer.vertex(0))->estimate();
}
void ICPORB_VO::Track(const cv::Mat& rgb, const cv::Mat& depth, const double& timestamp) {
  
  auto fICP = std::async(
    std::launch::async, 
    //prevent ambiguity of overload function by explicit casting
    (const ICP_VO::Pose&(ICP_VO::*)(const cv::Mat& , const double& ))(&ICP_VO::Track),
    pICPVO,
    std::cref(depth),
    std::cref(timestamp)
  );
  pORBVO->TrackRGBD(rgb, depth, timestamp);
  fICP.get();
}
ICPORB_VO::Poses ICPORB_VO::FusionTwoTrajectory() {
  g2o::SparseOptimizer optimizer;
  auto linear_solver = std::make_unique< g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType>>();
  auto block_solver = std::make_unique< SlamBlockSolver>(std::move(linear_solver));
  g2o::OptimizationAlgorithmGaussNewton * solver = new g2o::OptimizationAlgorithmGaussNewton( std::move(block_solver));
  optimizer.setAlgorithm(solver);
  {
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setEstimate(Pose::Identity());
    v->setId(0);
    v->setFixed(true);
    optimizer.addVertex(v);
  }
  Poses orbPoses;
  std::vector<std::string> icpTimes, orbTimes;
  std::string orbTrajFilename = "ORBTraj.txt";
  pORBVO->SaveTrajectoryTUM(orbTrajFilename);
  TUMTrajectoryReader(orbTrajFilename, orbPoses, orbTimes);

  Poses icpPoses(pICPVO->GetPoses().size());
  for (int i = 0; i < icpPoses.size(); ++i) {
    Pose tmp;
    tmp.matrix()=  pICPVO->GetPoses().at(i);
    icpPoses[i]= tmp;
  }
  // LinearSolverCSparse;
  {
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setEstimate(icpPoses[0]);
    v->setId(0);
    v->setFixed(true);
    optimizer.addVertex(v);
  }
  for (int i = 1; i < icpPoses.size(); ++i) {
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId(i);
    v->setEstimate(icpPoses[i]);
    optimizer.addVertex(v);

    g2o::EdgeSE3* e = new g2o::EdgeSE3();
    auto from = optimizer.vertex(i-1);
    auto to = optimizer.vertex(i);
    e->vertices()[0] = from;
    e->vertices()[1] = to;
    e->setMeasurement(icpPoses[i-1].inverse() * icpPoses[i]);
    e->setInformation(Eigen::MatrixX<number_t>::Identity(6,6));
    optimizer.addEdge(e);
  }
  if (orbEdgeLength != -1) {
    for (int i = orbEdgeLength; i < orbPoses.size(); ++i) {
      g2o::EdgeSE3* e = new g2o::EdgeSE3();
      auto from = optimizer.vertex(i - orbEdgeLength);
      auto to = optimizer.vertex(i);
      e->vertices()[0] = from;
      e->vertices()[1] = to;
      e->setMeasurement(orbPoses[i - orbEdgeLength].inverse() * orbPoses[i]);
      e->setInformation(Eigen::MatrixX<number_t>::Identity(6,6));
      optimizer.addEdge(e);
    }
  }
  
  optimizer.setVerbose(true);
  std::cerr << "Optimizing" << std::endl;
  optimizer.initializeOptimization();
  optimizer.optimize(4);
  std::cerr << "done." << std::endl;
  Poses out;
  for (int i = 0; i < icpPoses.size(); ++i) {
    out.push_back( dynamic_cast<g2o::VertexSE3*> (optimizer.vertex(i))->estimate());
  }
  // optimizer.save("");
  optimizer.clear();
  return out;
}
void ICPORB_VO::TUMTrajectoryReader(const std::string& file, Poses& out, std::vector<std::string>& times)
{
   
  char * line = NULL;
  size_t len = 0;
  FILE* pf = fopen(file.c_str(), "r");

  assert(pf && "opening file failed");
  std::string str_tmp(1000, '\0');
  number_t x, y, z, qx, qy, qz, qw;

  while (-1 != getline(&line, &len, pf)) {
    int n = sscanf(line, "%s %lf %lf %lf %lf %lf %lf %lf", &str_tmp[0], &x, &y, &z, &qx, &qy, &qz, &qw);
    assert(n == 8 && "read trajectory file failed");

    Eigen::Quaternion<number_t> q(qw, qx, qy, qz);
    g2o::Isometry3 p(q) ;
    p.translation() << x, y, z;
    out.push_back(p);
    std::string tmp = str_tmp;
    tmp.erase(tmp.begin()+ tmp.find('\0'),tmp.end());
    times.push_back(tmp);
  }
}
void ICPORB_VO::IncrementalTrack(const cv::Mat& rgb, const cv::Mat& depth, const ICP_VO::Cloud& cloud, const double& timestamp) {
  
  auto fORB = std::async(
    std::launch::async,
    &ORB_SLAM2::System::TrackRGBD,
    pORBVO,
    std::cref(rgb),
    std::cref(depth),
    std::cref(timestamp)
  );
  auto fICP = std::async(
    std::launch::async, 
    //prevent ambiguity of overload function by explicit casting
    (const ICP_VO::Pose&(ICP_VO::*)(const ICP_VO::Cloud& , const double& ))(&ICP_VO::Track),
    pICPVO,
    std::cref(cloud),
    std::cref(timestamp)
  );

  // //get ORB pose
  auto TORB = fORB.get();
  Pose PORB_inv  = Pose::Identity();
  //convert cv Mat to Pose
  bool IsValidORB = true;
  if (TORB.cols == 4 && TORB.rows == 4) {
    PORB_inv(0, 0) = TORB.at<float>(0, 0);
    PORB_inv(0, 1) = TORB.at<float>(0, 1);
    PORB_inv(0, 2) = TORB.at<float>(0, 2);
    PORB_inv(0, 3) = TORB.at<float>(0, 3);
    PORB_inv(1, 0) = TORB.at<float>(1, 0);
    PORB_inv(1, 1) = TORB.at<float>(1, 1);
    PORB_inv(1, 2) = TORB.at<float>(1, 2);
    PORB_inv(1, 3) = TORB.at<float>(1, 3);
    PORB_inv(2, 0) = TORB.at<float>(2, 0);
    PORB_inv(2, 1) = TORB.at<float>(2, 1);
    PORB_inv(2, 2) = TORB.at<float>(2, 2);
    PORB_inv(2, 3) = TORB.at<float>(2, 3);
    PORB_inv(3, 0) = TORB.at<float>(3, 0);
    PORB_inv(3, 1) = TORB.at<float>(3, 1);
    PORB_inv(3, 2) = TORB.at<float>(3, 2);
    PORB_inv(3, 3) = TORB.at<float>(3, 3);
    IsValidORB=true;
  } else {
    IsValidORB=false;
  }
  
  //get ICP pose
  auto TICP = fICP.get();

  //convert Matrix4d to Isometry
  Pose PICP(TICP);

  //fuse both trajectory
  posesORB.push_back(PORB_inv.inverse());
  posesICP.push_back(PICP);

  int n = 2;
  if (posesORB.size() < n) {
    n = posesORB.size();
  }
  if (posesICP.size() == 1) {
    currentPose = Pose::Identity();
    lastPose = Pose::Identity();
  } else {
    Fuse(n, IsValidORB, pICPVO->IsValid());
  }
}
void ICPORB_VO::IncrementalTrack(const cv::Mat& rgb, const cv::Mat& depth, const double& timestamp) {
  
  auto fORB = std::async(
    std::launch::async,
    &ORB_SLAM2::System::TrackRGBD,
    pORBVO,
    std::cref(rgb),
    std::cref(depth),
    std::cref(timestamp)
  );
  auto fICP = std::async(
    std::launch::async, 
    //prevent ambiguity of overload function by explicit casting
    (const ICP_VO::Pose&(ICP_VO::*)(const cv::Mat& , const double& ))(&ICP_VO::Track),
    pICPVO,
    std::cref(depth),
    std::cref(timestamp)
  );

  // //get ORB pose
  auto TORB = fORB.get();
  Pose PORB_inv  = Pose::Identity();
  //convert cv Mat to Pose
  bool IsValidORB = true;
  if (TORB.cols == 4 && TORB.rows == 4) {
    PORB_inv(0, 0) = TORB.at<float>(0, 0);
    PORB_inv(0, 1) = TORB.at<float>(0, 1);
    PORB_inv(0, 2) = TORB.at<float>(0, 2);
    PORB_inv(0, 3) = TORB.at<float>(0, 3);
    PORB_inv(1, 0) = TORB.at<float>(1, 0);
    PORB_inv(1, 1) = TORB.at<float>(1, 1);
    PORB_inv(1, 2) = TORB.at<float>(1, 2);
    PORB_inv(1, 3) = TORB.at<float>(1, 3);
    PORB_inv(2, 0) = TORB.at<float>(2, 0);
    PORB_inv(2, 1) = TORB.at<float>(2, 1);
    PORB_inv(2, 2) = TORB.at<float>(2, 2);
    PORB_inv(2, 3) = TORB.at<float>(2, 3);
    PORB_inv(3, 0) = TORB.at<float>(3, 0);
    PORB_inv(3, 1) = TORB.at<float>(3, 1);
    PORB_inv(3, 2) = TORB.at<float>(3, 2);
    PORB_inv(3, 3) = TORB.at<float>(3, 3);
    IsValidORB=true;
  } else {
    IsValidORB=false;
  }
  
  //get ICP pose
  auto TICP = fICP.get();

  //convert Matrix4d to Isometry
  Pose PICP(TICP);

  //fuse both trajectory
  posesORB.push_back(PORB_inv.inverse());
  posesICP.push_back(PICP);

  int n = 2;
  if (posesORB.size() < n) {
    n = posesORB.size();
  }
  if (posesICP.size() == 1) {
    currentPose = Pose::Identity();
    lastPose = Pose::Identity();
  } else {
    Fuse(n, IsValidORB, pICPVO->IsValid());
  }
}
