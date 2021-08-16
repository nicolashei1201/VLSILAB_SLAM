/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.


* File Name : testEigenProduct.cpp

* Purpose :

* Creation Date : 2020-08-23

* Last Modified : Sun Aug 23 12:28:20 2020

* Created By : Ji-Ying, Li 

*_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.*/
#include <Eigen/Core>
#include <iostream>

int main(int argc, char *argv[])
{
  Eigen::MatrixXd m = Eigen::MatrixXd::Ones(3,10);
  
  std::cout << "m" << std::endl;
  std::cout << m << std::endl;
  std::cout << m.colwise() + Eigen::Vector3d::Ones(3) << std::endl;
  return 0;
}
