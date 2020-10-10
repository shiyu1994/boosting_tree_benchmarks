LightGBM/lightgbm config=lightgbm.conf data=../data/higgs.train objective=binary 2>&1 | tee lightgbm_higgs_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/msltr.train objective=regression 2>&1 | tee lightgbm_msltr_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/yahoo.train objective=regression 2>&1 | tee lightgbm_yahoo_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/dataexpo_onehot.train objective=binary  2>&1 | tee lightgbm_dataexpo_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/allstate.train objective=binary num_leaves=127 learning_rate=0.02 2>&1 | tee lightgbm_allstate_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/adult.train objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=1,3,5,6,7,8,9,13 2>&1 | tee lightgbm_adult_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/amazon.train objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=0,1,2,3,4,5,6,7,8 2>&1 | tee lightgbm_amazon_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/appetency.train objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228 2>&1 | tee lightgbm_appetency_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/click.train objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=1,2,3,6,7,8,9,10 2>&1 | tee lightgbm_click_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/internet.train objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=0,1,2,11,12,18,19,20,21,31,32,33,34,36,37,38,39,59,60,61,62 2>&1 | tee lightgbm_internet_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/kick.train objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=0,1,3,4,5,6,7,8,9,10,12,13,14,23,24,25,26,27,29,31,32,33,34 2>&1 | tee lightgbm_kick_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/upselling.train objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228 2>&1 | tee lightgbm_upselling_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/nips_b.train objective=binary cateogrical_feature=0,1,2,3,4,6,7,8,10,17,18,19,20,21,22,23,24 2>&1 | tee lightgbm_nips_b_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/nips_c.train objective=binary cateogrical_feature=0,1,2,4,5,7,8,9,12,14,15,18,19,20,21,22,24,26,28,30,31,33,34,36,39,41,42,43,44,46,47,48,49,50,51,52,53,54,56,57,58,59,60 2>&1 | tee lightgbm_nips_c_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/year.train objective=binary 2>&1 | tee lightgbm_year_speed.log

