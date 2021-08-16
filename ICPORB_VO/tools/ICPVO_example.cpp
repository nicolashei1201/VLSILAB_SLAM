/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.


* File Name : ICPVO_example.cpp

* Purpose :

* Creation Date : 2020-08-29

* Last Modified : 廿廿年八月廿九日 (週六) 十一時二分十五秒

* Created By :  

*_._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._._.*/

#include <ICP_VO.h>
#include <Eigen/Geometry>
#include <fstream>

void LoadImagePaths(const std::string& tum_association_filepath, std::vector<std::string>& vstr_rgb_filepath, std::vector<std::string>& vstr_depth_filepath, std::vector<std::string>& vstr_timestamp);

int main(int argc, char *argv[])
{
  //parse comment
  if (argc != 5) {
    std::cout << "example: ./build/ICPVO_example path_to_association_file path_to_sequence path_to_setting path_to_output_path" << std::endl;
    return 1;
  }

  //access association file
  std::vector<std::string> vstr_rgb_filepath;
  std::vector<std::string> vstr_depth_filepath;
  std::vector<std::string> vstr_timestamp;
  LoadImagePaths(argv[1], vstr_rgb_filepath, vstr_depth_filepath, vstr_timestamp);
  
  //open output trajectory file
  std::ofstream fout(argv[4]);
  fout.close();
  fout.open(argv[4], std::fstream::app);

  //instance VO 
  ICP_VO vo( argv[3]);
  
  for (int i = 0; i < vstr_rgb_filepath.size(); ++i) {
    //Read image
    const std::string rgb_path =std::string(argv[2])+'/'+vstr_rgb_filepath[i];
    const std::string depth_path = std::string(argv[2])+'/'+vstr_depth_filepath[i];
    cv::Mat target_color = cv::imread(rgb_path, 1);
    cv::Mat target_depth = cv::imread(depth_path, -1);

    vo.Track(target_depth, std::stod(vstr_timestamp[i]));
    Eigen::Isometry3d current_pose_iso;
    current_pose_iso= vo.GetPoses().back();
    Eigen::Quaterniond q(current_pose_iso.rotation());
    fout << vstr_timestamp[i] << " " << current_pose_iso(0,3) 
                            << " " << current_pose_iso(1,3)
                            << " " << current_pose_iso(2,3) 
                            << " " << q.x() << " " << q.y()
                            << " " << q.z() << " " << q.w() << std::endl;

// #ifdef OUTPUT_TUM_POSE
    //show pose on screen
    std::cout << vstr_timestamp[i] << " " << current_pose_iso(0,3) 
                            << " " << current_pose_iso(1,3)
                            << " " << current_pose_iso(2,3) 
                            << " " << q.x() << " " << q.y()
                            << " " << q.z() << " " << q.w() << std::endl;

// #endif
  }
  fout.close();
  return 0;
}

void LoadImagePaths(const std::string& tum_association_filepath, std::vector<std::string>& vstr_rgb_filepath, std::vector<std::string>& vstr_depth_filepath, std::vector<std::string>& vstr_timestamp)
{
  std::ifstream f_association(tum_association_filepath);
  while (!f_association.eof()) {
    std::string s;
    getline(f_association,s);
    if (!s.empty()) {
      std::stringstream ss(s);
      std::string t;
      std::string sRGB, sD;
      ss >> t;
      ss >> sRGB;
      vstr_rgb_filepath.push_back(std::move(sRGB));
      ss >> t;
      vstr_timestamp.push_back(t);
      ss >> sD;
      vstr_depth_filepath.push_back(std::move(sD));
    }
  }
}
