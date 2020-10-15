cd tune/lightgbm
python gen_bach.py
./tune.sh
cd ../..

cd tune/xgboost
python gen_bach.py
./tune.sh
cd ../..

cd tune/catboost
python gen_bach.py
./tune.sh
cd ../..