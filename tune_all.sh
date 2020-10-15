#!/bin/bash

cd tune/lightgbm
python gen_bash.py $1 $2
./tune.sh
cd ../..

cd tune/xgboost
python gen_bash.py $1 $2
./tune.sh
cd ../..

cd tune/catboost
python gen_bash.py $1 $2
./tune.sh
cd ../..
