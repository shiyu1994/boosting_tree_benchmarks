#!/bin/bash

cd tune/lightgbm
python gen_bach.py $1 $2
./tune.sh
cd ../..

cd tune/xgboost
python gen_bach.py $1 $2
./tune.sh
cd ../..

cd tune/catboost
python gen_bach.py $1 $2
./tune.sh
cd ../..