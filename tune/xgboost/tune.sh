python -u ../xgboost_exact_tuner.py ../../data/higgs_small.train ../../data/higgs_small.test ../../data/higgs.cd tmp binary 100 5 1000 16 higgs_exact.log > higgs_exact_tune.log
python -u ../xgboost_exact_tuner.py ../../data/yahoo_small.train ../../data/yahoo_small.test ../../data/yahoo.cd tmp ranking 100 5 1000 16 yahoo_exact.log ../../data/yahoo_small.train.query ../../data/yahoo_small.test.query > yahoo_exact_tune.log
python -u ../xgboost_exact_tuner.py ../../data/msltr_small.train ../../data/msltr_small.test ../../data/msltr.cd tmp ranking 100 5 1000 16 msltr_exact.log ../../data/msltr_small.train.query ../../data/msltr_small.test.query > msltr_exact_tune.log
python -u ../xgboost_exact_tuner.py ../../data/dataexpo_onehot_small.train ../../data/dataexpo_onehot_small.test ../../data/dataexpo_onehot.cd tmp binary 100 5 1000 16 expo_exact.log > expo_exact_tune.log
python -u ../xgboost_exact_tuner.py ../../data/allstate_small.train ../../data/allstate_small.test ../../data/allstate.cd tmp binary 100 5 1000 16 allstate_exact.log > allstate_exact_tune.log
python -u ../xgboost_exact_tuner.py ../../data/adult.train ../../data/adult.test ../../data/adult.cd tmp binary 100 5 1000 16 adult_exact.log > adult_exact_tune.log
python -u ../xgboost_exact_tuner.py ../../data/amazon.train ../../data/amazon.test ../../data/amazon.cd tmp binary 100 5 1000 16 amazon_exact.log > amazon_exact_tune.log
python -u ../xgboost_exact_tuner.py ../../data/appetency.train ../../data/appetency.test ../../data/appetency.cd tmp binary 100 5 1000 16 appetency_exact.log > appetency_exact_tune.log
python -u ../xgboost_exact_tuner.py ../../data/internet.train ../../data/internet.test ../../data/internet.cd tmp binary 100 5 1000 16 internet_exact.log > internet_exact_tune.log
python -u ../xgboost_exact_tuner.py ../../data/upselling.train ../../data/upselling.test ../../data/upselling.cd tmp binary 100 5 1000 16 upselling_exact.log > upselling_exact_tune.log
python -u ../xgboost_exact_tuner.py ../../data/click.train ../../data/click.test ../../data/click.cd tmp binary 100 5 1000 16 click_exact.log > click_exact_tune.log
python -u ../xgboost_exact_tuner.py ../../data/kick.train ../../data/kick.test ../../data/kick.cd tmp binary 100 5 1000 16 kick_exact.log > kick_exact_tune.log
python -u ../xgboost_exact_tuner.py ../../data/nips_b.train ../../data/nips_b.test ../../data/nips_b.cd tmp binary 100 5 1000 16 nips_b_exact.log > nips_b_exact_tune.log
python -u ../xgboost_exact_tuner.py ../../data/nips_c.train ../../data/nips_c.test ../../data/nips_c.cd tmp binary 100 5 1000 16 nips_c_exact.log > nips_c_exact_tune.log
python -u ../xgboost_exact_tuner.py ../../data/year.train ../../data/year.test ../../data/year.cd tmp regression 100 5 1000 16 year_exact.log > year_exact_tune.log


python -u ../xgboost_hist_tuner.py ../../data/higgs_small.train ../../data/higgs_small.test ../../data/higgs.cd tmp binary 100 5 1000 16 higgs_hist.log > higgs_hist_tune.log
python -u ../xgboost_hist_tuner.py ../../data/yahoo_small.train ../../data/yahoo_small.test ../../data/yahoo.cd tmp ranking 100 5 1000 16 yahoo_hist.log ../../data/yahoo_small.train.query ../../data/yahoo_small.test.query > yahoo_hist_tune.log
python -u ../xgboost_hist_tuner.py ../../data/msltr_small.train ../../data/msltr_small.test ../../data/msltr.cd tmp ranking 100 5 1000 16 msltr_hist.log ../../data/msltr_small.train.query ../../data/msltr_small.test.query > msltr_hist_tune.log
python -u ../xgboost_hist_tuner.py ../../data/dataexpo_onehot_small.train ../../data/dataexpo_onehot_small.test ../../data/dataexpo_onehot.cd tmp binary 100 5 1000 16 expo_hist.log > expo_hist_tune.log
python -u ../xgboost_hist_tuner.py ../../data/allstate_small.train ../../data/allstate_small.test ../../data/allstate.cd tmp binary 100 5 1000 16 allstate_hist.log > allstate_hist_tune.log
python -u ../xgboost_hist_tuner.py ../../data/adult.train ../../data/adult.test ../../data/adult.cd tmp binary 100 5 1000 16 adult_hist.log > adult_hist_tune.log
python -u ../xgboost_hist_tuner.py ../../data/amazon.train ../../data/amazon.test ../../data/amazon.cd tmp binary 100 5 1000 16 amazon_hist.log > amazon_hist_tune.log
python -u ../xgboost_hist_tuner.py ../../data/appetency.train ../../data/appetency.test ../../data/appetency.cd tmp binary 100 5 1000 16 appetency_hist.log > appetency_hist_tune.log
python -u ../xgboost_hist_tuner.py ../../data/internet.train ../../data/internet.test ../../data/internet.cd tmp binary 100 5 1000 16 internet_hist.log > internet_hist_tune.log
python -u ../xgboost_hist_tuner.py ../../data/upselling.train ../../data/upselling.test ../../data/upselling.cd tmp binary 100 5 1000 16 upselling_hist.log > upselling_hist_tune.log
python -u ../xgboost_hist_tuner.py ../../data/click.train ../../data/click.test ../../data/click.cd tmp binary 100 5 1000 16 click_hist.log > click_hist_tune.log
python -u ../xgboost_hist_tuner.py ../../data/kick.train ../../data/kick.test ../../data/kick.cd tmp binary 100 5 1000 16 kick_hist.log > kick_hist_tune.log
python -u ../xgboost_hist_tuner.py ../../data/nips_b.train ../../data/nips_b.test ../../data/nips_b.cd tmp binary 100 5 1000 16 nips_b_hist.log > nips_b_hist_tune.log
python -u ../xgboost_hist_tuner.py ../../data/nips_c.train ../../data/nips_c.test ../../data/nips_c.cd tmp binary 100 5 1000 16 nips_c_hist.log > nips_c_hist_tune.log
python -u ../xgboost_hist_tuner.py ../../data/year.train ../../data/year.test ../../data/year.cd tmp regression 100 5 1000 16 year_hist.log > year_hist_tune.log
