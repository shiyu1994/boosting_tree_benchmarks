deactivate
rm -rf .env_tune
virtualenv .env_tune --python=python3
source .env_tune/bin/activate
pip install numpy sklearn pandas scipy matplotlib lightgbm xgboost catboost hyperopt

cd tune/lightgbm
./tune.sh
cd ..

cd tune/xgboost
./tune.sh
cd ..

cd tune/catboost
./tune.sh
cd ..

deactivate
rm -rf .env_tune