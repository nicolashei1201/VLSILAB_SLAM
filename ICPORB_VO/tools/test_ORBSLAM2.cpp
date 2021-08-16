/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.


* File Name : test_ORBSLAM2.cpp

* Purpose :

* Creation Date : 2020-08-19

* Last Modified : Mon Aug 24 22:29:29 2020

* Created By : Ji-Ying, Li 

*_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.*/

#include <opencv2/opencv.hpp>
#include "System.h"

int main(int argc, char *argv[])
{
  
  ORB_SLAM2::System orb_tracker("", "", ORB_SLAM2::System::RGBD, true);
  return 0;
}
