rm -rf LightGBM
git clone --recursive https://github.com/Microsoft/LightGBM.git
cd LightGBM
mkdir build
cd build
cmake ..
make -j
cd ../python-package
python setup.py install
