LightGBM/lightgbm config=lightgbm.conf data=../data/higgs.train valid=../data/higgs.test objective=binary metric=auc 2>&1 | tee lightgbm_higgs_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/msltr.train valid=../data/msltr.test objective=lambdarank metric=ndcg 2>&1 | tee lightgbm_msltr_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/yahoo.train valid=../data/yahoo.test objective=lambdarank metric=ndcg 2>&1 | tee lightgbm_yahoo_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/dataexpo_onehot.train valid=../data/dataexpo_onehot.test metric=auc objective=binary 2>&1 | tee lightgbm_dataexpo_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/allstate.train valid=../data/allstate.test metric=auc objective=binary num_leaves=127 learning_rate=0.02 2>&1 | tee lightgbm_allstate_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/adult.train valid=../data/adult.test metric=auc objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=2,4,6,7,8,9,10,14 2>&1 | tee lightgbm_adult_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/amazon.train valid=../data/amazon.test metric=auc objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=1,2,3,4,5,6,7,8,9 2>&1 | tee lightgbm_amazon_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/appetency.train valid=../data/appetency.test metric=auc objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229 2>&1 | tee lightgbm_appetency_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/click.train valid=../data/click.test metric=auc objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=2,3,4,7,8,9,10,11 2>&1 | tee lightgbm_click_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/internet.train valid=../data/internet.test metric=auc objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=1,2,3,12,13,19,20,21,22,32,33,34,35,37,38,39,40,60,61,62,63 2>&1 | tee lightgbm_internet_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/kick.train valid=../data/kick.test metric=auc objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=1,2,4,5,6,7,8,9,10,11,13,14,15,24,25,26,27,28,30,32,33,34,35 2>&1 | tee lightgbm_kick_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/upselling.train valid=../data/upselling.test metric=auc objective=binary num_leaves=127 learning_rate=0.02 cateogrical_feature=191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229 2>&1 | tee lightgbm_upselling_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/nips_b.train valid=../data/nips_b.test metric=auc objective=binary cateogrical_feature=1,2,3,4,5,7,8,9,11,18,19,20,21,22,23,24,25 2>&1 | tee lightgbm_nips_b_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/nips_c.train valid=../data/nips_c.test metric=auc objective=binary cateogrical_feature=1,2,3,5,6,8,9,10,13,15,16,19,20,21,22,23,25,27,29,31,32,34,35,37,40,42,43,44,45,47,48,49,50,51,52,53,54,55,57,58,59,60,61 2>&1 | tee lightgbm_nips_c_accuracy.log

LightGBM/lightgbm config=lightgbm.conf data=../data/year.train valid=../data/year.test objective=binary metric=auc 2>&1 | tee lightgbm_year_accuracy.log
