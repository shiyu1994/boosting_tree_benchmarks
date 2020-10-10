rm -rf xgboost
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; mkdir build; cd build; cmake ..; make -j
