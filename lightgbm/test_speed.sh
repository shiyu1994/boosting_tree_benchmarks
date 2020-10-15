LightGBM/lightgbm config=lightgbm.conf data=../data/higgs.train objective=binary 2>&1 | tee lightgbm_higgs_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/msltr.train objective=regression 2>&1 | tee lightgbm_msltr_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/yahoo.train objective=regression 2>&1 | tee lightgbm_yahoo_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/dataexpo_onehot.train objective=binary  2>&1 | tee lightgbm_dataexpo_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/allstate.train objective=binary num_leaves=127 learning_rate=0.02 2>&1 | tee lightgbm_allstate_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/adult.train objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=2,4,6,7,8,9,10,14 2>&1 | tee lightgbm_adult_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/amazon.train objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=1,2,3,4,5,6,7,8,9 2>&1 | tee lightgbm_amazon_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/appetency.train objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229 2>&1 | tee lightgbm_appetency_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/click.train objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=2,3,4,7,8,9,10,11 2>&1 | tee lightgbm_click_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/internet.train objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=1,2,3,12,13,19,20,21,22,32,33,34,35,37,38,39,40,60,61,62,63 2>&1 | tee lightgbm_internet_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/kick.train objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=1,2,4,5,6,7,8,9,10,11,13,14,15,24,25,26,27,28,30,32,33,34,35 2>&1 | tee lightgbm_kick_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/upselling.train objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229 2>&1 | tee lightgbm_upselling_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/nips_b.train objective=binary cateogrical_feature=1,2,3,4,5,7,8,9,11,18,19,20,21,22,23,24,25 2>&1 | tee lightgbm_nips_b_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/nips_c.train objective=binary cateogrical_feature=1,2,3,5,6,8,9,10,13,15,16,19,20,21,22,23,25,27,29,31,32,34,35,37,40,42,43,44,45,47,48,49,50,51,52,53,54,55,57,58,59,60,61 2>&1 | tee lightgbm_nips_c_speed.log

LightGBM/lightgbm config=lightgbm.conf data=../data/year.train objective=binary 2>&1 | tee lightgbm_year_speed.log

