python -u ../catboost_symmetric_tuner.py ../../data/higgs_small.train.cat ../../data/higgs_small.test.cat ../../data/higgs.train.cd tmp binary 100 5 1000 16 higgs.log cat 2>&1 > higgs_tune.log
#python -u ../catboost_symmetric_tuner.py ../../data/yahoo_small.train.cat ../../data/yahoo_small.test.cat ../../data/yahoo.train.cd tmp ranking 100 5 1000 16 yahoo.log cat 2>&1 > yahoo_tune.log
#python -u ../catboost_symmetric_tuner.py ../../data/msltr_small.train.cat ../../data/msltr_small.test.cat ../../data/msltr.train.cd tmp ranking 100 5 1000 16 msltr.log cat 2>&1 > msltr_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/dataexpo_onehot_small.train.cat ../../data/dataexpo_onehot_small.test.cat ../../data/dataexpo_onehot.train.cd tmp binary 100 5 1000 16 expo.log libsvm 2>&1 > expo_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/allstate_small.train.cat ../../data/allstate_small.test.cat ../../data/allstate.train.cd tmp binary 100 5 1000 16 allstate.log libsvm 2>&1 > allstate_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/adult.train.cat ../../data/adult.test.cat ../../data/adult.train.cd tmp binary 100 5 1000 16 adult.log cat 2>&1 > adult_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/amazon.train.cat ../../data/amazon.test.cat ../../data/amazon.train.cd tmp binary 100 5 1000 16 amazon.log cat 2>&1 > amazon_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/appetency.train.cat ../../data/appetency.test.cat ../../data/appetency.train.cd tmp binary 100 5 1000 16 appetency.log cat 2>&1 > appetency_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/internet.train.cat ../../data/internet.test.cat ../../data/internet.train.cd tmp binary 100 5 1000 16 internet.log cat 2>&1 > internet_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/upselling.train.cat ../../data/upselling.test.cat ../../data/upselling.train.cd tmp binary 100 5 1000 16 upselling.log cat 2>&1 > upselling_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/click.train.cat ../../data/click.test.cat ../../data/click.train.cd tmp binary 100 5 1000 16 click.log cat 2>&1 > click_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/kick.train.cat ../../data/kick.test.cat ../../data/kick.train.cd tmp binary 100 5 1000 16 kick.log cat 2>&1 > kick_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/nips_b.train.cat ../../data/nips_b.test.cat ../../data/nips_b.train.cd tmp binary 100 5 1000 16 nips_b.log libsvm 2>&1 > nips_b_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/nips_c.train.cat ../../data/nips_c.test.cat ../../data/nips_c.train.cd tmp binary 100 5 1000 16 nips_c.log libsvm 2>&1 > nips_c_tune.log
python -u ../catboost_symmetric_tuner.py ../../data/year.train.cat ../../data/year.test.cat ../../data/year.train.cd tmp regression 100 5 1000 16 year.log libsvm 2>&1 > year_tune.log

python -u ../catboost_leafwise_tuner.py ../../data/higgs_small.train.cat ../../data/higgs_small.test.cat ../../data/higgs.train.cd tmp binary 100 5 1000 16 higgs.log cat 2>&1 > higgs_tune.log
#python -u ../catboost_leafwise_tuner.py ../../data/yahoo_small.train.cat ../../data/yahoo_small.test.cat ../../data/yahoo.train.cd tmp ranking 100 5 1000 16 yahoo.log cat 2>&1 > yahoo_tune.log
#python -u ../catboost_leafwise_tuner.py ../../data/msltr_small.train.cat ../../data/msltr_small.test.cat ../../data/msltr.train.cd tmp ranking 100 5 1000 16 msltr.log cat 2>&1 > msltr_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/dataexpo_onehot_small.train.cat ../../data/dataexpo_onehot_small.test.cat ../../data/dataexpo_onehot.train.cd tmp binary 100 5 1000 16 expo.log libsvm 2>&1 > expo_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/allstate_small.train.cat ../../data/allstate_small.test.cat ../../data/allstate.train.cd tmp binary 100 5 1000 16 allstate.log libsvm 2>&1 > allstate_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/adult.train.cat ../../data/adult.test.cat ../../data/adult.train.cd tmp binary 100 5 1000 16 adult.log cat 2>&1 > adult_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/amazon.train.cat ../../data/amazon.test.cat ../../data/amazon.train.cd tmp binary 100 5 1000 16 amazon.log cat 2>&1 > amazon_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/appetency.train.cat ../../data/appetency.test.cat ../../data/appetency.train.cd tmp binary 100 5 1000 16 appetency.log cat 2>&1 > appetency_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/internet.train.cat ../../data/internet.test.cat ../../data/internet.train.cd tmp binary 100 5 1000 16 internet.log cat 2>&1 > internet_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/upselling.train.cat ../../data/upselling.test.cat ../../data/upselling.train.cd tmp binary 100 5 1000 16 upselling.log cat 2>&1 > upselling_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/click.train.cat ../../data/click.test.cat ../../data/click.train.cd tmp binary 100 5 1000 16 click.log cat 2>&1 > click_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/kick.train.cat ../../data/kick.test.cat ../../data/kick.train.cd tmp binary 100 5 1000 16 kick.log cat 2>&1 > kick_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/nips_b.train.cat ../../data/nips_b.test.cat ../../data/nips_b.train.cd tmp binary 100 5 1000 16 nips_b.log libsvm 2>&1 > nips_b_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/nips_c.train.cat ../../data/nips_c.test.cat ../../data/nips_c.train.cd tmp binary 100 5 1000 16 nips_c.log libsvm 2>&1 > nips_c_tune.log
python -u ../catboost_leafwise_tuner.py ../../data/year.train.cat ../../data/year.test.cat ../../data/year.train.cd tmp regression 100 5 1000 16 year.log libsvm 2>&1 > year_tune.log
