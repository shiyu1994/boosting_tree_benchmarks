rm -rf catboost
git clone --recursive https://github.com/catboost/catboost.git
export CC=/usr/bin/clang-7
export CXX=/usr/bin/clang++-7
cd catboost
make -j -f make/app.CLANG7-LINUX-X86_64.makefile