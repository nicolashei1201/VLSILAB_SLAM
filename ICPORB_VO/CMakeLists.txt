cmake_minimum_required(VERSION 3.2)
project(ICPORB_VO)

#
SET(CMAKE_BUILD_TYPE Release)
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")
# SET(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -Wextra -march=native") # bug: without march native, the opencv imread funciton will cause memory corrupt
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -Wextra -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -Wextra ") # bug: without march native, the opencv imread funciton will cause memory corrupt
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -Wextra ")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
# add_definitions(-DCOMPILEDWITHC11)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake_modules/")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)


# find packages
#find_package(Eigen3 3.3.90  REQUIRED )
SET (EIGEN3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/Thirdparty/Eigen/eigen3)
find_package(Pangolin REQUIRED)
SET(OpenCV_DIR /home/jyli/packages/installation/OpenCV-4.1.1/lib/cmake/opencv4)

find_package( OpenCV 4 REQUIRED )
SET(G2O_ROOT /home/jjhu/Desktop/hei/jy_code/ICPORB_VO/Thirdparty/g2o)
find_package( G2O REQUIRED )
INCLUDE_DIRECTORIES(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/Thirdparty/ORB_SLAM2
  ${PROJECT_SOURCE_DIR}/Thirdparty/ORB_SLAM2/include
  #${PROJECT_SOURCE_DIR}/Thirdparty/g2o/include
  ${G2O_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIR}
)

# build ORB SLAM2
add_library(ORB_SLAM2 SHARED
  Thirdparty/ORB_SLAM2/src/System.cc
  Thirdparty/ORB_SLAM2/src/Tracking.cc
  Thirdparty/ORB_SLAM2/src/LocalMapping.cc
  Thirdparty/ORB_SLAM2/src/LoopClosing.cc
  Thirdparty/ORB_SLAM2/src/ORBextractor.cc
  Thirdparty/ORB_SLAM2/src/ORBmatcher.cc
  Thirdparty/ORB_SLAM2/src/FrameDrawer.cc
  Thirdparty/ORB_SLAM2/src/Converter.cc
  Thirdparty/ORB_SLAM2/src/MapPoint.cc
  Thirdparty/ORB_SLAM2/src/KeyFrame.cc
  Thirdparty/ORB_SLAM2/src/Map.cc
  Thirdparty/ORB_SLAM2/src/MapDrawer.cc
  Thirdparty/ORB_SLAM2/src/Optimizer.cc
  Thirdparty/ORB_SLAM2/src/PnPsolver.cc
  Thirdparty/ORB_SLAM2/src/Frame.cc
  Thirdparty/ORB_SLAM2/src/KeyFrameDatabase.cc
  Thirdparty/ORB_SLAM2/src/Sim3Solver.cc
  Thirdparty/ORB_SLAM2/src/Initializer.cc
  Thirdparty/ORB_SLAM2/src/Viewer.cc
)
set (G2O_LIBS 
# Find the core elements
${G2O_STUFF_LIBRARY}
${G2O_CORE_LIBRARY}
# Find the CLI library 
${G2O_CLI_LIBRARY}
# Find the pluggable solvers 
#${G2O_SOLVER_CHOLMOD}
${G2O_SOLVER_CSPARSE}
${G2O_SOLVER_CSPARSE_EXTENSION}
${G2O_SOLVER_DENSE}
${G2O_SOLVER_PCG}
${G2O_SOLVER_SLAM2D_LINEAR}
${G2O_SOLVER_STRUCTURE_ONLY}
${G2O_SOLVER_EIGEN}
# Find the predefined types  
${G2O_TYPES_DATA}
${G2O_TYPES_ICP}
${G2O_TYPES_SBA}
${G2O_TYPES_SCLAM2D}
${G2O_TYPES_SIM3}
${G2O_TYPES_SLAM2D}
${G2O_TYPES_SLAM3D}
)
target_link_libraries( ORB_SLAM2
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
# ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o*.so
${G2O_LIBS}
)
# build ICP_ORB

add_library(ICPORB_VO SHARED
  src/EAS_ICP.cpp
  src/ICP_VO.cpp
  src/ICPORB_VO.cpp
)
target_link_libraries( ICPORB_VO 
  ORB_SLAM2
)

# Build executable files
add_executable(test_CVMat_copy
    ./tools/test_CVMat_copy.cpp
)
target_link_libraries(test_CVMat_copy ${OpenCV_LIBS} ) 

add_executable(test_ORBSLAM2
    ${PROJECT_SOURCE_DIR}/tools/test_ORBSLAM2.cpp
)
target_link_libraries(test_ORBSLAM2  ORB_SLAM2)

add_executable(test_EigeProduct
  ${PROJECT_SOURCE_DIR}/tools/testEigenProduct.cpp
)
target_link_libraries(test_EigeProduct  ORB_SLAM2)

add_executable(test_ORBG2O
  ${PROJECT_SOURCE_DIR}/tools/testORBG2O.cpp
)
target_link_libraries(test_ORBG2O  ORB_SLAM2)
add_executable( rgbd_tum_off
    ${PROJECT_SOURCE_DIR}/tools/rgbd_tum_off.cpp
)
target_link_libraries(rgbd_tum_off ICPORB_VO)
add_executable( rgbd_tum
    ${PROJECT_SOURCE_DIR}/tools/rgbd_tum.cpp
)
target_link_libraries(rgbd_tum ICPORB_VO)
add_executable(EASICP_example
  ${PROJECT_SOURCE_DIR}/tools/EASICP_example.cpp
)
target_link_libraries(EASICP_example  ICPORB_VO)
add_executable( ICPVO_example
  ${PROJECT_SOURCE_DIR}/tools/ICPVO_example.cpp
)
target_link_libraries(ICPVO_example ICPORB_VO)
add_executable( ICPORBVO_example
  ${PROJECT_SOURCE_DIR}/tools/ICPORBVO_example.cpp
)
target_link_libraries(ICPORBVO_example ICPORB_VO)
