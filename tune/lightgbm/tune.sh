python -u ../lightgbm_tuner.py ../../data/higgs_small.train ../../data/higgs_small.test ../../data/higgs.cd ../../data/higgs.count tmp binary 100 5 1000 16 higgs.log 2>&1 > higgs_tune.log
python -u ../lightgbm_tuner.py ../../data/yahoo_small.train ../../data/yahoo_small.test ../../data/yahoo.cd ../../data/yahoo.count tmp ranking 100 5 1000 16 yahoo.log ../../data/yahoo_small.train.query ../../data/yahoo_small.test.query 2>&1 > yahoo_tune.log
python -u ../lightgbm_tuner.py ../../data/msltr_small.train ../../data/msltr_small.test ../../data/msltr.cd ../../data/msltr.count tmp ranking 100 5 1000 16 msltr.log ../../data/msltr_small.train.query ../../data/msltr_small.test.query 2>&1 > msltr_tune.log
python -u ../lightgbm_tuner.py ../../data/dataexpo_onehot_small.train ../../data/dataexpo_onehot_small.test ../../data/dataexpo_onehot.cd ../../data/dataexpo_onehot.count tmp binary 100 5 1000 16 expo.log 2>&1 > expo_tune.log
python -u ../lightgbm_tuner.py ../../data/allstate_small.train ../../data/allstate_small.test ../../data/allstate.cd ../../data/allstate.count tmp binary 100 5 1000 16 allstate.log 2>&1 > allstate_tune.log
python -u ../lightgbm_tuner.py ../../data/adult.train ../../data/adult.test ../../data/adult.cd ../../data/adult.count tmp binary 100 5 1000 16 adult.log 2>&1 > adult_tune.log
python -u ../lightgbm_tuner.py ../../data/amazon.train ../../data/amazon.test ../../data/amazon.cd ../../data/amazon.count tmp binary 100 5 1000 16 amazon.log 2>&1 > amazon_tune.log
python -u ../lightgbm_tuner.py ../../data/appetency.train ../../data/appetency.test ../../data/appetency.cd ../../data/appetency.count tmp binary 100 5 1000 16 appetency.log 2>&1 > appetency_tune.log
python -u ../lightgbm_tuner.py ../../data/internet.train ../../data/internet.test ../../data/internet.cd ../../data/internet.count tmp binary 100 5 1000 16 internet.log 2>&1 > internet_tune.log
python -u ../lightgbm_tuner.py ../../data/upselling.train ../../data/upselling.test ../../data/upselling.cd ../../data/upselling.count tmp binary 100 5 1000 16 upselling.log 2>&1 > upselling_tune.log
python -u ../lightgbm_tuner.py ../../data/click.train ../../data/click.test ../../data/click.cd ../../data/click.count tmp binary 100 5 1000 16 click.log 2>&1 > click_tune.log
python -u ../lightgbm_tuner.py ../../data/kick.train ../../data/kick.test ../../data/kick.cd ../../data/kick.count tmp binary 100 5 1000 16 kick.log 2>&1 > kick_tune.log
python -u ../lightgbm_tuner.py ../../data/nips_b.train ../../data/nips_b.test ../../data/nips_b.cd ../../data/nips_b.count tmp binary 100 5 1000 16 nips_b.log 2>&1 > nips_b_tune.log
python -u ../lightgbm_tuner.py ../../data/nips_c.train ../../data/nips_c.test ../../data/nips_c.cd ../../data/nips_c.count tmp binary 100 5 1000 16 nips_c.log 2>&1 > nips_c_tune.log
python -u ../lightgbm_tuner.py ../../data/year.train ../../data/year.test ../../data/year.cd ../../data/year.count tmp regression 100 5 1000 16 year.log 2>&1 > year_tune.log
