catboost/catboost/app/catboost fit --params-file params_leaf_wise_large.json --learn-set ../data/higgs.train.cat --test-set ../data/higgs.test.cat --column-description ../data/higgs.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_leafwise_higgs_accuracy.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_large.json --learn-set ../data/msltr.train.cat --test-set ../data/msltr.test.cat --column-description ../data/msltr.train.cd  --loss-function YetiRank --eval-metric NDCG:top=1 2>&1 | tee catboost_leafwise_msltr_accuracy_1.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_large.json --learn-set ../data/yahoo.train.cat --test-set ../data/yahoo.test.cat --column-description ../data/yahoo.train.cd  --loss-function YetiRank --eval-metric NDCG:top=1 2>&1 | tee catboost_leafwise_yahoo_accuracy_1.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_large.json --learn-set ../data/msltr.train.cat --test-set ../data/msltr.test.cat --column-description ../data/msltr.train.cd  --loss-function YetiRank --eval-metric NDCG:top=3 2>&1 | tee catboost_leafwise_msltr_accuracy_3.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_large.json --learn-set ../data/yahoo.train.cat --test-set ../data/yahoo.test.cat --column-description ../data/yahoo.train.cd  --loss-function YetiRank --eval-metric NDCG:top=3 2>&1 | tee catboost_leafwise_yahoo_accuracy_3.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_large.json --learn-set ../data/msltr.train.cat --test-set ../data/msltr.test.cat --column-description ../data/msltr.train.cd  --loss-function YetiRank --eval-metric NDCG:top=5 2>&1 | tee catboost_leafwise_msltr_accuracy_5.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_large.json --learn-set ../data/yahoo.train.cat --test-set ../data/yahoo.test.cat --column-description ../data/yahoo.train.cd  --loss-function YetiRank --eval-metric NDCG:top=5 2>&1 | tee catboost_leafwise_yahoo_accuracy_5.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_large.json --learn-set ../data/msltr.train.cat --test-set ../data/msltr.test.cat --column-description ../data/msltr.train.cd  --loss-function YetiRank --eval-metric NDCG:top=10 2>&1 | tee catboost_leafwise_msltr_accuracy_10.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_large.json --learn-set ../data/yahoo.train.cat --test-set ../data/yahoo.test.cat --column-description ../data/yahoo.train.cd  --loss-function YetiRank --eval-metric NDCG:top=10 2>&1 | tee catboost_leafwise_yahoo_accuracy_10.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_large.json --learn-set libsvm://../data/dataexpo_onehot.train.cat --test-set libsvm://../data/dataexpo_onehot.test.cat --column-description ../data/dataexpo_onehot.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_leafwise_dataexpo_onehot_accuracy.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_small.json --learn-set libsvm://../data/allstate.train.cat --test-set libsvm://../data/allstate.test.cat --column-description ../data/allstate.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_leafwise_allstate_accuracy.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_small.json --learn-set ../data/adult.train.cat --test-set ../data/adult.test.cat --column-description ../data/adult.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_leafwise_adult_accuracy.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_small.json --learn-set ../data/amazon.train.cat --test-set ../data/amazon.test.cat --column-description ../data/amazon.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_leafwise_amazon_accuracy.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_small.json --learn-set ../data/appetency.train.cat --test-set ../data/appetency.test.cat --column-description ../data/appetency.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_leafwise_appetency_accuracy.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_small.json --learn-set ../data/click.train.cat --test-set ../data/click.test.cat --column-description ../data/click.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_leafwise_click_accuracy.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_small.json --learn-set ../data/internet.train.cat --test-set ../data/internet.test.cat --column-description ../data/internet.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_leafwise_internet_accuracy.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_small.json --learn-set ../data/kick.train.cat --test-set ../data/kick.test.cat --column-description ../data/kick.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_leafwise_kick_accuracy.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_small.json --learn-set ../data/upselling.train.cat --test-set ../data/upselling.test.cat --column-description ../data/upselling.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_leafwise_upselling_accuracy.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_large.json --learn-set libsvm://../data/nips_b.train.cat --test-set libsvm://../data/nips_b.test.cat --column-description ../data/nips_b.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_leafwise_nips_b_accuracy.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_large.json --learn-set libsvm://../data/nips_c.train.cat --test-set libsvm://../data/nips_c.test.cat --column-description ../data/nips_c.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_leafwise_nips_c_accuracy.log

catboost/catboost/app/catboost fit --params-file params_leaf_wise_large.json --learn-set libsvm://../data/year.train.cat --test-set libsvm://../data/year.test.cat --column-description ../data/year.train.cd  --loss-function RMSE --eval-metric RMSE 2>&1 | tee catboost_leafwise_year_accuracy.log




catboost/catboost/app/catboost fit --params-file params_symmetric_large.json --learn-set ../data/higgs.train.cat --test-set ../data/higgs.test.cat --column-description ../data/higgs.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_symmetric_higgs_accuracy.log

catboost/catboost/app/catboost fit --params-file params_symmetric_large.json --learn-set ../data/msltr.train.cat --test-set ../data/msltr.test.cat --column-description ../data/msltr.train.cd  --loss-function YetiRank --eval-metric NDCG:top=1 2>&1 | tee catboost_symmetric_msltr_accuracy_1.log

catboost/catboost/app/catboost fit --params-file params_symmetric_large.json --learn-set ../data/yahoo.train.cat --test-set ../data/yahoo.test.cat --column-description ../data/yahoo.train.cd  --loss-function YetiRank --eval-metric NDCG:top=1 2>&1 | tee catboost_symmetric_yahoo_accuracy_1.log

catboost/catboost/app/catboost fit --params-file params_symmetric_large.json --learn-set ../data/msltr.train.cat --test-set ../data/msltr.test.cat --column-description ../data/msltr.train.cd  --loss-function YetiRank --eval-metric NDCG:top=3 2>&1 | tee catboost_symmetric_msltr_accuracy_3.log

catboost/catboost/app/catboost fit --params-file params_symmetric_large.json --learn-set ../data/yahoo.train.cat --test-set ../data/yahoo.test.cat --column-description ../data/yahoo.train.cd  --loss-function YetiRank --eval-metric NDCG:top=3 2>&1 | tee catboost_symmetric_yahoo_accuracy_3.log

catboost/catboost/app/catboost fit --params-file params_symmetric_large.json --learn-set ../data/msltr.train.cat --test-set ../data/msltr.test.cat --column-description ../data/msltr.train.cd  --loss-function YetiRank --eval-metric NDCG:top=5 2>&1 | tee catboost_symmetric_msltr_accuracy_5.log

catboost/catboost/app/catboost fit --params-file params_symmetric_large.json --learn-set ../data/yahoo.train.cat --test-set ../data/yahoo.test.cat --column-description ../data/yahoo.train.cd  --loss-function YetiRank --eval-metric NDCG:top=5 2>&1 | tee catboost_symmetric_yahoo_accuracy_5.log

catboost/catboost/app/catboost fit --params-file params_symmetric_large.json --learn-set ../data/msltr.train.cat --test-set ../data/msltr.test.cat --column-description ../data/msltr.train.cd  --loss-function YetiRank --eval-metric NDCG:top=10 2>&1 | tee catboost_symmetric_msltr_accuracy_10.log

catboost/catboost/app/catboost fit --params-file params_symmetric_large.json --learn-set ../data/yahoo.train.cat --test-set ../data/yahoo.test.cat --column-description ../data/yahoo.train.cd  --loss-function YetiRank --eval-metric NDCG:top=10 2>&1 | tee catboost_symmetric_yahoo_accuracy_10.log

catboost/catboost/app/catboost fit --params-file params_symmetric_large.json --learn-set libsvm://../data/dataexpo_onehot.train.cat --test-set libsvm://../data/dataexpo_onehot.test.cat --column-description ../data/dataexpo_onehot.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_symmetric_dataexpo_onehot_accuracy.log

catboost/catboost/app/catboost fit --params-file params_symmetric_small.json --learn-set libsvm://../data/allstate.train.cat --test-set libsvm://../data/allstate.test.cat --column-description ../data/allstate.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_symmetric_allstate_accuracy.log

catboost/catboost/app/catboost fit --params-file params_symmetric_small.json --learn-set ../data/adult.train.cat --test-set ../data/adult.test.cat --column-description ../data/adult.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_symmetric_adult_accuracy.log

catboost/catboost/app/catboost fit --params-file params_symmetric_small.json --learn-set ../data/amazon.train.cat --test-set ../data/amazon.test.cat --column-description ../data/amazon.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_symmetric_amazon_accuracy.log

catboost/catboost/app/catboost fit --params-file params_symmetric_small.json --learn-set ../data/appetency.train.cat --test-set ../data/appetency.test.cat --column-description ../data/appetency.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_symmetric_appetency_accuracy.log

catboost/catboost/app/catboost fit --params-file params_symmetric_small.json --learn-set ../data/click.train.cat --test-set ../data/click.test.cat --column-description ../data/click.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_symmetric_click_accuracy.log

catboost/catboost/app/catboost fit --params-file params_symmetric_small.json --learn-set ../data/internet.train.cat --test-set ../data/internet.test.cat --column-description ../data/internet.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_symmetric_internet_accuracy.log

catboost/catboost/app/catboost fit --params-file params_symmetric_small.json --learn-set ../data/kick.train.cat --test-set ../data/kick.test.cat --column-description ../data/kick.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_symmetric_kick_accuracy.log

catboost/catboost/app/catboost fit --params-file params_symmetric_small.json --learn-set ../data/upselling.train.cat --test-set ../data/upselling.test.cat --column-description ../data/upselling.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_symmetric_upselling_accuracy.log

catboost/catboost/app/catboost fit --params-file params_symmetric_large.json --learn-set libsvm://../data/nips_b.train.cat --test-set libsvm://../data/nips_b.test.cat --column-description ../data/nips_b.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_symmetric_nips_b_accuracy.log

catboost/catboost/app/catboost fit --params-file params_symmetric_large.json --learn-set libsvm://../data/nips_c.train.cat --test-set libsvm://../data/nips_c.test.cat --column-description ../data/nips_c.train.cd  --loss-function Logloss --eval-metric AUC 2>&1 | tee catboost_symmetric_nips_c_accuracy.log

catboost/catboost/app/catboost fit --params-file params_symmetric_large.json --learn-set libsvm://../data/year.train.cat --test-set libsvm://../data/year.test.cat --column-description ../data/year.train.cd  --loss-function RMSE --eval-metric RMSE 2>&1 | tee catboost_symmetric_year_accuracy.log