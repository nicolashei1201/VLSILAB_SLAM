# #Installing Pangolin
cd Thirdparty
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build
cd build
cmake ../ -DAVFORMAT_INCLUDE_DIR="" -DCPP11_NO_BOOST=ON
make -j
cd ../..
#
# # build DBoW2
cd DBoW2/
mkdir build
cd build
cmake ..
make -j
cd ../..

#build g2o
cd g2o/
mkdir build
cd build 
cmake ..
make -j && make install -j
cd ../../..

# #build ICPORB VO
mkdir build
cd build
cmake ..
make -j

cd ..

