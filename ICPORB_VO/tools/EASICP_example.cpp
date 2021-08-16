/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.


* File Name : EASICP_example.cpp

* Purpose :

* Creation Date : 2020-08-29

* Last Modified : 廿廿年八月廿九日 (週六) 十時36分50秒

* Created By :  

*_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.*/

// #include <EAS_ICP.h>
#include <ICP_VO.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
  if (argc != 4) {
    std::cout << "./build/rgbd_tum path_to_source_depth_file path_to_target_depth_file path_to_setting " << std::endl;
    return 1;
  }
  ICP_VO icp_vo(argv[3]);
  
  cv::Mat source_depth = cv::imread(argv[1], -1);
  cv::Mat target_depth = cv::imread(argv[2], -1);
  icp_vo.Track(source_depth, 0);
  icp_vo.Track(target_depth, 1);
  
  std::cout << icp_vo.GetPoses().back() << std::endl;

  return 0;
}
