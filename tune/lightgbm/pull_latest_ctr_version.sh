rm -rf LightGBM
git clone --recursive https://github.com/shiyu1994/LightGBM.git
cd LightGBM
git checkout ctr
git submodule update --recursive --init
mkdir build
cd build
cmake ..
make -j
cd ../python-package
python setup.py install
