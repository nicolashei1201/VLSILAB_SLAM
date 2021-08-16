/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.


* File Name : test_CVMat_copy.cpp

* Purpose :

* Creation Date : 2020-08-12

* Last Modified : Wed Aug 19 15:32:39 2020

* Created By : Ji-Ying, Li 

*_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.*/

#include <iostream>
#include <opencv2/opencv.hpp>


void test_mat(const unsigned char* test)
{
  cv::Mat mat1(10,10, CV_8UC1,(unsigned char*) test);

  std::cout << mat1 << std::endl;
  std::cout << "test addr:" << (void*)test << std::endl;
  std::cout << "mat1 data addr:" << (void*)mat1.data << std::endl;
  std::cout << mat1 << std::endl;
  
}
int main(int argc, char *argv[])
{

  
  unsigned char test[100] = {0};

  test_mat(test);
  
  return 0;
}
