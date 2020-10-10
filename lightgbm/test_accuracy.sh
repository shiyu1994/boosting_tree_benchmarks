LightGBM/lightgbm config=lightgbm.conf data=../data/higgs.train valid=../data/higgs.test objective=binary metric=auc 2>&1 | tee lightgbm_higgs_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/msltr.train valid=../data/msltr.test objective=lambdarank metric=ndcg 2>&1 | tee lightgbm_msltr_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/yahoo.train valid=../data/yahoo.test objective=lambdarank metric=ndcg 2>&1 | tee lightgbm_yahoo_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/dataexpo_onehot.train valid=../data/dataexpo_onehot.test metric=auc objective=binary 2>&1 | tee lightgbm_dataexpo_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/allstate.train valid=../data/allstate.test metric=auc objective=binary num_leaves=127 learning_rate=0.02 2>&1 | tee lightgbm_allstate_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/adult.train valid=../data/adult.test metric=auc objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=1,3,5,6,7,8,9,13 2>&1 | tee lightgbm_adult_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/amazon.train valid=../data/amazon.test metric=auc objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=0,1,2,3,4,5,6,7,8 2>&1 | tee lightgbm_amazon_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/appetency.train valid=../data/appetency.test metric=auc objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228 2>&1 | tee lightgbm_appetency_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/click.train valid=../data/click.test metric=auc objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=1,2,3,6,7,8,9,10 2>&1 | tee lightgbm_click_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/internet.train valid=../data/internet.test metric=auc objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=0,1,2,11,12,18,19,20,21,31,32,33,34,36,37,38,39,59,60,61,62 2>&1 | tee lightgbm_internet_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/kick.train valid=../data/kick.test metric=auc objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=0,1,3,4,5,6,7,8,9,10,12,13,14,23,24,25,26,27,29,31,32,33,34 2>&1 | tee lightgbm_kick_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/upselling.train valid=../data/upselling.test metric=auc objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228 2>&1 | tee lightgbm_upselling_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/nips_b.train valid=../data/nips_b.test metric=auc objective=binary cateogrical_feature=0,1,2,3,4,6,7,8,10,17,18,19,20,21,22,23,24 2>&1 | tee lightgbm_nips_b_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/nips_c.train valid=../data/nips_c.test metric=auc objective=binary cateogrical_feature=0,1,2,4,5,7,8,9,12,14,15,18,19,20,21,22,24,26,28,30,31,33,34,36,39,41,42,43,44,46,47,48,49,50,51,52,53,54,56,57,58,59,60 2>&1 | tee lightgbm_nips_c_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/year.train valid=../data/year.test objective=binary metric=auc 2>&1 | tee lightgbm_year_accuracy.log





