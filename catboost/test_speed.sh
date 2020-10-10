catboost/catboost/app/catboost fit --params-file params_leaf_wise_large.json --learn-set ../data/higgs.train.cat --column-description ../data/higgs.train.cd  --loss-function Logloss 2>&1 | tee catboost_leafwise_higgs_speed.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_large.json --learn-set ../data/msltr.train.cat --column-description ../data/msltr.train.cd  --loss-function YetiRank 2>&1 | tee catboost_leafwise_msltr_speed.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_large.json --learn-set ../data/yahoo.train.cat --column-description ../data/yahoo.train.cd  --loss-function YetiRank 2>&1 | tee catboost_leafwise_yahoo_speed.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_large.json --learn-set libsvm://../data/dataexpo_onehot.train.cat --column-description ../data/dataexpo_onehot.train.cd  --loss-function Logloss 2>&1 | tee catboost_leafwise_dataexpo_onehot_speed.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_small.json --learn-set libsvm://../data/allstate.train.cat --column-description ../data/allstate.train.cd  --loss-function Logloss 2>&1 | tee catboost_leafwise_allstate_speed.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_small.json --learn-set ../data/adult.train.cat --column-description ../data/adult.train.cd  --loss-function Logloss  2>&1 | tee catboost_leafwise_adult_speed.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_small.json --learn-set ../data/amazon.train.cat --column-description ../data/amazon.train.cd  --loss-function Logloss  2>&1 | tee catboost_leafwise_amazon_speed.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_small.json --learn-set ../data/appetency.train.cat --column-description ../data/appetency.train.cd  --loss-function Logloss  2>&1 | tee catboost_leafwise_appetency_speed.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_small.json --learn-set ../data/click.train.cat --column-description ../data/click.train.cd  --loss-function Logloss  2>&1 | tee catboost_leafwise_click_speed.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_small.json --learn-set ../data/internet.train.cat --column-description ../data/internet.train.cd  --loss-function Logloss  2>&1 | tee catboost_leafwise_internet_speed.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_small.json --learn-set ../data/kick.train.cat --column-description ../data/kick.train.cd  --loss-function Logloss  2>&1 | tee catboost_leafwise_kick_speed.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_small.json --learn-set ../data/upselling.train.cat --column-description ../data/upselling.train.cd  --loss-function Logloss  2>&1 | tee catboost_leafwise_upselling_speed.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_large.json --learn-set libsvm://../data/nips_b.train.cat --column-description ../data/nips_b.train.cd  --loss-function Logloss  2>&1 | tee catboost_leafwise_nips_b_speed.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_large.json --learn-set libsvm://../data/nips_c.train.cat --column-description ../data/nips_c.train.cd  --loss-function Logloss  2>&1 | tee catboost_leafwise_nips_c_speed.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_large.json --learn-set libsvm://../data/year.train.cat --column-description ../data/year.train.cd  --loss-function RMSE 2>&1 | tee catboost_leafwise_year_speed.log




catboost/catboost/app/catboost fit --params-file params_symmetric_large.json --learn-set ../data/higgs.train.cat --test-set ../data/higgs.test.cat --column-description ../data/higgs.train.cd  --loss-function Logloss  2>&1 | tee catboost_symmetric_higgs_speed.log

catboost/catboost/app/catboost fit --params-file params_symmetric_large.json --learn-set ../data/msltr.train.cat --test-set ../data/msltr.test.cat --column-description ../data/msltr.train.cd  --loss-function YetiRank 2>&1 | tee catboost_symmetric_msltr_speed.log

catboost/catboost/app/catboost fit --params-file params_symmetric_large.json --learn-set ../data/yahoo.train.cat --test-set ../data/yahoo.test.cat --column-description ../data/yahoo.train.cd  --loss-function YetiRank 2>&1 | tee catboost_symmetric_yahoo_speed.log

catboost/catboost/app/catboost fit --params-file params_symmetric_large.json --learn-set libsvm://../data/dataexpo_onehot.train.cat --test-set libsvm://../data/dataexpo_onehot.test.cat --column-description ../data/dataexpo_onehot.train.cd  --loss-function Logloss  2>&1 | tee catboost_symmetric_dataexpo_onehot_speed.log

catboost/catboost/app/catboost fit --params-file params_symmetric_small.json --learn-set libsvm://../data/allstate.train.cat --test-set libsvm://../data/allstate.test.cat --column-description ../data/allstate.train.cd  --loss-function Logloss  2>&1 | tee catboost_symmetric_allstate_speed.log

catboost/catboost/app/catboost fit --params-file params_symmetric_small.json --learn-set ../data/adult.train.cat --test-set ../data/adult.test.cat --column-description ../data/adult.train.cd  --loss-function Logloss  2>&1 | tee catboost_symmetric_adult_speed.log

catboost/catboost/app/catboost fit --params-file params_symmetric_small.json --learn-set ../data/amazon.train.cat --test-set ../data/amazon.test.cat --column-description ../data/amazon.train.cd  --loss-function Logloss  2>&1 | tee catboost_symmetric_amazon_speed.log

catboost/catboost/app/catboost fit --params-file params_symmetric_small.json --learn-set ../data/appetency.train.cat --test-set ../data/appetency.test.cat --column-description ../data/appetency.train.cd  --loss-function Logloss  2>&1 | tee catboost_symmetric_appetency_speed.log

catboost/catboost/app/catboost fit --params-file params_symmetric_small.json --learn-set ../data/click.train.cat --test-set ../data/click.test.cat --column-description ../data/click.train.cd  --loss-function Logloss  2>&1 | tee catboost_symmetric_click_speed.log

catboost/catboost/app/catboost fit --params-file params_symmetric_small.json --learn-set ../data/internet.train.cat --test-set ../data/internet.test.cat --column-description ../data/internet.train.cd  --loss-function Logloss  2>&1 | tee catboost_symmetric_internet_speed.log

catboost/catboost/app/catboost fit --params-file params_symmetric_small.json --learn-set ../data/kick.train.cat --test-set ../data/kick.test.cat --column-description ../data/kick.train.cd  --loss-function Logloss  2>&1 | tee catboost_symmetric_kick_speed.log

catboost/catboost/app/catboost fit --params-file params_symmetric_small.json --learn-set ../data/upselling.train.cat --test-set ../data/upselling.test.cat --column-description ../data/upselling.train.cd  --loss-function Logloss  2>&1 | tee catboost_symmetric_upselling_speed.log

catboost/catboost/app/catboost fit --params-file params_symmetric_large.json --learn-set libsvm://../data/nips_b.train.cat --test-set libsvm://../data/nips_b.test.cat --column-description ../data/nips_b.train.cd  --loss-function Logloss  2>&1 | tee catboost_symmetric_nips_b_speed.log

catboost/catboost/app/catboost fit --params-file params_symmetric_large.json --learn-set libsvm://../data/nips_c.train.cat --test-set libsvm://../data/nips_c.test.cat --column-description ../data/nips_c.train.cd  --loss-function Logloss  2>&1 | tee catboost_symmetric_nips_c_speed.log

catboost/catboost/app/catboost fit --params-file params_symmetric_large.json --learn-set libsvm://../data/year.train.cat --test-set libsvm://../data/year.test.cat --column-description ../data/year.train.cd  --loss-function RMSE --eval-metric RMSE 2>&1 | tee catboost_symmetric_year_speed.log