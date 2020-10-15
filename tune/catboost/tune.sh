python -u ../catboost_symmetric_tuner.py ../../data/higgs_small.train ../../data/higgs_small.test ../../data/higgs.cd tmp binary 100 5 1000 16 higgs_symmetric.log cat 2>&1 > higgs_symmetric_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/yahoo_small.train ../../data/yahoo_small.test ../../data/yahoo.cd tmp ranking 100 5 1000 16 yahoo_symmetric.log libsvm ../../data/yahoo_small.train.query ../../data/yahoo_small.test.query 2>&1 > yahoo_symmetric_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/msltr_small.train ../../data/msltr_small.test ../../data/msltr.cd tmp ranking 100 5 1000 16 msltr_symmetric.log libsvm ../../data/msltr_small.train.query ../../data/msltr_small.test.query 2>&1 > msltr_symmetric_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/dataexpo_onehot_small.train ../../data/dataexpo_onehot_small.test ../../data/dataexpo_onehot.cd tmp binary 100 5 1000 16 expo_symmetric.log libsvm 2>&1 > expo_symmetric_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/allstate_small.train ../../data/allstate_small.test ../../data/allstate.cd tmp binary 100 5 1000 16 allstate_symmetric.log libsvm 2>&1 > allstate_symmetric_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/adult.train ../../data/adult.test ../../data/adult.cd tmp binary 100 5 1000 16 adult_symmetric.log cat 2>&1 > adult_symmetric_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/amazon.train ../../data/amazon.test ../../data/amazon.cd tmp binary 100 5 1000 16 amazon_symmetric.log cat 2>&1 > amazon_symmetric_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/appetency.train ../../data/appetency.test ../../data/appetency.cd tmp binary 100 5 1000 16 appetency_symmetric.log cat 2>&1 > appetency_symmetric_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/internet.train ../../data/internet.test ../../data/internet.cd tmp binary 100 5 1000 16 internet_symmetric.log cat 2>&1 > internet_symmetric_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/upselling.train ../../data/upselling.test ../../data/upselling.cd tmp binary 100 5 1000 16 upselling_symmetric.log cat 2>&1 > upselling_symmetric_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/click.train ../../data/click.test ../../data/click.cd tmp binary 100 5 1000 16 click_symmetric.log cat 2>&1 > click_symmetric_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/kick.train ../../data/kick.test ../../data/kick.cd tmp binary 100 5 1000 16 kick_symmetric.log cat 2>&1 > kick_symmetric_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/nips_b.train ../../data/nips_b.test ../../data/nips_b.cd tmp binary 100 5 1000 16 nips_b_symmetric.log libsvm 2>&1 > nips_b_symmetric_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/nips_c.train ../../data/nips_c.test ../../data/nips_c.cd tmp binary 100 5 1000 16 nips_c_symmetric.log libsvm 2>&1 > nips_c_symmetric_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/year.train ../../data/year.test ../../data/year.cd tmp regression 100 5 1000 16 year_symmetric.log libsvm 2>&1 > year_symmetric_tune.log

python -u ../catboost_leafwise_tuner.py ../../data/higgs_small.train ../../data/higgs_small.test ../../data/higgs.cd tmp binary 100 5 1000 16 higgs_hist.log cat 2>&1 > higgs_hist_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/yahoo_small.train ../../data/yahoo_small.test ../../data/yahoo.cd tmp ranking 100 5 1000 16 yahoo_hist.log libsvm ../../data/yahoo_small.train.query ../../data/yahoo_small.test.query 2>&1 > yahoo_hist_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/msltr_small.train ../../data/msltr_small.test ../../data/msltr.cd tmp ranking 100 5 1000 16 msltr_hist.log libsvm ../../data/msltr_small.train.query ../../data/msltr_small.test.query 2>&1 > msltr_hist_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/dataexpo_onehot_small.train ../../data/dataexpo_onehot_small.test ../../data/dataexpo_onehot.cd tmp binary 100 5 1000 16 expo_hist.log libsvm 2>&1 > expo_hist_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/allstate_small.train ../../data/allstate_small.test ../../data/allstate.cd tmp binary 100 5 1000 16 allstate_hist.log libsvm 2>&1 > allstate_hist_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/adult.train ../../data/adult.test ../../data/adult.cd tmp binary 100 5 1000 16 adult_hist.log cat 2>&1 > adult_hist_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/amazon.train ../../data/amazon.test ../../data/amazon.cd tmp binary 100 5 1000 16 amazon_hist.log cat 2>&1 > amazon_hist_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/appetency.train ../../data/appetency.test ../../data/appetency.cd tmp binary 100 5 1000 16 appetency_hist.log cat 2>&1 > appetency_hist_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/internet.train ../../data/internet.test ../../data/internet.cd tmp binary 100 5 1000 16 internet_hist.log cat 2>&1 > internet_hist_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/upselling.train ../../data/upselling.test ../../data/upselling.cd tmp binary 100 5 1000 16 upselling_hist.log cat 2>&1 > upselling_hist_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/click.train ../../data/click.test ../../data/click.cd tmp binary 100 5 1000 16 click_hist.log cat 2>&1 > click_hist_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/kick.train ../../data/kick.test ../../data/kick.cd tmp binary 100 5 1000 16 kick_hist.log cat 2>&1 > kick_hist_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/nips_b.train ../../data/nips_b.test ../../data/nips_b.cd tmp binary 100 5 1000 16 nips_b_hist.log libsvm 2>&1 > nips_b_hist_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/nips_c.train ../../data/nips_c.test ../../data/nips_c.cd tmp binary 100 5 1000 16 nips_c_hist.log libsvm 2>&1 > nips_c_hist_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/year.train ../../data/year.test ../../data/year.cd tmp regression 100 5 1000 16 year_hist.log libsvm 2>&1 > year_hist_tune.log
