/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.


* File Name : testORBG2O.cpp

* Purpose :

* Creation Date : 2020-08-23

* Last Modified : Mon Aug 24 22:29:05 2020

* Created By : Ji-Ying, Li 

*_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.*/

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/types/slam3d/edge_se3.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/core/factory.h"
#include "g2o/core/base_vertex.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

int main(int argc, char *argv[])
{
  
  using Pose = Eigen::Transform<double, 3, Eigen::Isometry, Eigen::RowMajor>;
  g2o::SparseOptimizer optimizer;
  using  SlamBlockSolver=g2o::BlockSolver<g2o::BlockSolverTraits<-1,-1>>;
  auto linear_solver = std::make_unique<g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType>>();
  auto block_solver = std::make_unique< SlamBlockSolver>(std::move(linear_solver));
  g2o::OptimizationAlgorithmGaussNewton * solver = new g2o::OptimizationAlgorithmGaussNewton( std::move(block_solver));
  optimizer.setAlgorithm(solver);
  // g2o::VertexSE3* v = new g2o::VertexSE3();
  g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
  //v->setEstimate(Pose::Identity());
  //v->setId(0);
  //v->setFixed(true);
  //optimizer.addVertex(v);
  g2o::VertexSE3* v = new g2o::VertexSE3();
  // delete v;
  return 0;
}
