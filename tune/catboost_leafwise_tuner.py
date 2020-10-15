from tuner import Tuner
import catboost as cat
import hyperopt as hpt
from hyperopt import hp
import numpy as np
import sys

class CatBoostTuner(Tuner):
    def __init__(self, train_fname, test_fname, feat_types_fname, work_dir, task, max_evals,\
            n_cv_folds, num_trees, num_threads, time_log_fname, data_format, train_query_fname=None, test_query_fname=None):
        super().__init__(train_fname, test_fname, feat_types_fname, work_dir, task, max_evals,\
            n_cv_folds, time_log_fname, train_query_fname=train_query_fname, test_query_fname=test_query_fname)
        print("using CatBoost in " + str(cat.__file__))
        if self.task == "ranking":
            self.objective = "YetiRank"
        elif self.task == "binary":
            self.objective = "Logloss"
        elif self.task == "regression":
            self.objective = "RMSE"
        else:
            raise NotImplementedError("Unknown task " + self.task)
        self.num_threads = num_threads
        self.num_trees = num_trees
        param_dict = {
            'learning_rate': hp.loguniform('learning_rate', -7, 0),
            'num_leaves' : hp.qloguniform('num_leaves', 1, 7, 1),
            'border_count': 2 ** (hp.randint('max_bin_minus_4', 7) + 4),
            'random_strength': hp.randint('random_strength', 20) + 1,
            'colsample_bylevel': hp.uniform('feature_fraction', 0.5, 1),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'bagging_temperature': hp.uniform('bagging_temperature', 0.0, 1.0),
            'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
            'l2_leaf_reg': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_log', -16, 2)]),
            "grow_policy":"Lossguide",
            "counter_calc_method":"SkipTest",
            "boosting_type":"Plain"
        }
        if len(self.categorical_features) > 0:
            param_dict["one_hot_max_size"] = hp.randint('one_hot_max_size', 26)
            param_dict["max_ctr_complexity"] = hp.randint('max_ctr_complexity', 6) + 1
        self.param_space = param_dict
        self.column_description_fname = feat_types_fname
        self.data_format = data_format


    def fullfill_parameters(self, params, seed):
        if self.objective == "Logloss":
            self.metric = "AUC"
        elif self.objective == "RMSE":
            self.metric = "RMSE"
        elif self.objective == "YetiRank":
            self.metric = "NDCG:top=5;type=Exp"
        else:
            raise NotImplementedError("unknown objective type {}".format(self.objective))

        params.update({
            "loss_function": self.objective,
            "eval_metric": self.metric,
            "thread_count": self.num_threads,
            'sampling_frequency': 'PerTree',
            "random_seed":seed,
            "iterations": self.num_trees
        })

        params["num_leaves"] = int(params["num_leaves"])
        params["min_data_in_leaf"] = int(params["min_data_in_leaf"])

    def _get_group_id_from_file(self, query_fname):
        if query_fname is not None:
            group_id = []
            with open(query_fname, "r") as in_file:
                cur_group_id = 0
                for line in in_file:
                    item_counts = int(line.strip())
                    for _ in range(item_counts):
                        group_id += [cur_group_id]
                    cur_group_id += 1
            return np.array(group_id)
        else:
            return None

    def eval(self, params, train_file, test_file, seed=0, train_query_fname=None, test_query_fname=None):
        prefix = ""
        if self.data_format == "libsvm":
            prefix = self.data_format + "://"
        train_group_id = self._get_group_id_from_file(train_query_fname)
        test_group_id = self._get_group_id_from_file(test_query_fname)
        train_data = cat.Pool(prefix + train_file, column_description=self.column_description_fname)
        test_data = cat.Pool(prefix + test_file, column_description=self.column_description_fname)
        train_data.set_group_id(train_group_id)
        test_data.set_group_id(test_group_id)
        self.fullfill_parameters(params, seed)
        print("eval with params " + str(params))
        cat_booster = cat.CatBoost(params=params)
        cat_booster.fit(train_data, eval_set=[test_data], verbose=1)
        eval_results = cat_booster.get_evals_result()
        results = eval_results["validation"][self.metric]
        train_data = None
        test_data = None
        return cat_booster, results

if __name__ == "__main__":
    if len(sys.argv) != 12 and len(sys.argv) != 14:
        print("usage: python catboost_leafwise_tuner.py <train_fname> <test_fname> <feat_types_fname>"
            "<work_dir> <task> <max_evals> <n_cv_folds> <num_trees> <num_threads> <time_log_fname> <data_format> [train_query_fname test_query_fname]")
        exit(0)
    train_fname = sys.argv[1]
    test_fname = sys.argv[2]
    feat_types_fname = sys.argv[3]
    work_dir = sys.argv[4]
    task = sys.argv[5]
    max_evals = int(sys.argv[6])
    n_cv_folds = int(sys.argv[7])
    num_trees = int(sys.argv[8])
    num_threads = int(sys.argv[9])
    time_log_fname = sys.argv[10]
    data_format = sys.argv[11]
    if len(sys.argv) == 12:
        catboost_tuner = CatBoostTuner(train_fname, test_fname, feat_types_fname, work_dir, task, max_evals,\
            n_cv_folds, num_trees, num_threads, time_log_fname, data_format)
    else:
        train_query_fname = sys.argv[12]
        test_query_fname = sys.argv[13]
        catboost_tuner = CatBoostTuner(train_fname, test_fname, feat_types_fname, work_dir, task, max_evals, n_cv_folds, num_trees,\
            num_threads, time_log_fname, data_format, train_query_fname=train_query_fname, test_query_fname=test_query_fname)
    catboost_tuner.run()