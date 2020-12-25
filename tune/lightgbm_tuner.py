from tuner import Tuner
import lightgbm as lgb
import hyperopt as hpt
from hyperopt import hp
import numpy as np
import sys

class LightGBMTuner(Tuner):
    def __init__(self, train_fname, test_fname, feat_types_fname, cat_count_fname, work_dir, task,\
            max_evals, n_cv_folds, num_trees, num_threads, time_log_fname, cat_type, train_query_fname=None, test_query_fname=None):
        super().__init__(train_fname, test_fname, feat_types_fname, work_dir, task,\
            max_evals, n_cv_folds, time_log_fname, train_query_fname=train_query_fname, test_query_fname=test_query_fname)
        print("using LightGBM in " + str(lgb.__file__))
        if self.task == "ranking":
            self.objective = "lambdarank"
        else:
            self.objective = self.task
        self.num_threads = num_threads
        self.num_trees = num_trees
        self.cat_type = cat_type
        param_dict = {
            'learning_rate': hp.loguniform('learning_rate', -7, 0),
            'num_leaves' : hp.qloguniform('num_leaves', 1, 7, 1),
            'max_depth':0,
            'max_bin': 2 ** (hp.randint('max_bin_minus_4', 7) + 4),
            'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
            'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
            'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', -16, 5),
            'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_log', -16, 2)]),
            'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_log', -16, 2)]),
            "tree_learner":"serial"
        }
        if len(self.categorical_features) > 0:
            max_cat_count = 0
            with open(cat_count_fname, "r") as in_file:
                for line in in_file:
                    _, cat_count = line.strip().split("\t")
                    cat_count = int(cat_count)
                    if cat_count > max_cat_count:
                        max_cat_count = cat_count
            if self.cat_type == "old":
                param_dict["cat_smooth"] = hp.qloguniform('cat_smooth', 0, 8, 1)
                param_dict["cat_l2"] = hp.qloguniform('cat_l2', 0, 6, 1)
                param_dict["max_cat_threshold"] = hp.randint('max_cat_threshold', max_cat_count // 2) + 1
            elif self.cat_type == "new":
                param_dict["cat_converters"] = hp.choice("ctr_method", ["ctr,count", "ctr"])
                param_dict["prior_weight"] = hp.qloguniform('prior_weight', 0, 8, 1)
                param_dict["num_ctr_folds"] = hp.randint('num_ctr_folds', 12) + 3
        self.param_space = param_dict
        self.metric = None


    def fullfill_parameters(self, params, seed):
        if self.objective == "binary":
            self.metric = "auc"
        elif self.objective == "regression":
            self.metric = "rmse"
        elif self.objective == "lambdarank":
            self.metric = "ndcg"
            params.update({"ndcg_eval_at":5})
        else:
            raise NotImplementedError("unknown objective type {}".format(self.objective))

        params.update({
            "objective": self.objective,
            "metric": self.metric,
            "num_threads": self.num_threads,
            "bagging_freq": 1,
            "seed":seed,
            "num_trees": self.num_trees
        })

        params["num_leaves"] = int(params["num_leaves"])
        params["min_data_in_leaf"] = int(params["min_data_in_leaf"])
        if "max_cat_threshold" in params:
            params["max_cat_threshold"] = int(params["max_cat_threshold"])

    def eval(self, params, train_file, test_file, seed=0, train_query_fname=None, test_query_fname=None,\
        early_stopping_rounds=None, num_rounds=None):
        if train_query_fname is not None:
            assert test_query_fname is not None
            train_group = np.genfromtxt(train_query_fname, delimiter=",", dtype=np.int)
            test_group = np.genfromtxt(test_query_fname, delimiter=",", dtype=np.int)
        else:
            train_group = None
            test_group = None
        if "cat_converters" in params:
            cat_converters = params["cat_converters"]
        else:
            cat_converters = "raw"
        print("train_file = ", train_file)
        print("test_file = ", test_file)
        train_data = lgb.Dataset(train_file, group=train_group, cat_converters=cat_converters)
        test_data = lgb.Dataset(test_file, reference=train_data, group=test_group, cat_converters=cat_converters)
        eval_results = {}
        self.fullfill_parameters(params, seed)
        print("eval with params " + str(params))
        lgb_booster = lgb.train(params, train_data, valid_sets=[test_data], valid_names=["test"], evals_result=eval_results,
            categorical_feature=self.categorical_features, keep_training_booster=True, early_stopping_rounds=early_stopping_rounds)
        for key in eval_results:
            print("key=", key)
        if "ndcg_eval_at" in params:
            results = eval_results["test"][self.metric + "@{}".format(params["ndcg_eval_at"])]
        else:
            results = eval_results["test"][self.metric]
        train_data = None
        test_data = None
        return lgb_booster, results

if __name__ == "__main__":
    if len(sys.argv) != 13 and len(sys.argv) != 15:
        print("usage: python lightgbm_tuner.py <train_fname> <test_fname> <feat_types_fname> <cat_count_fname>"
            "<work_dir> <task> <max_evals> <n_cv_folds> <num_trees> <num_threads> <time_log_fname> <cat_type> [train_query_fname test_query_fname]")
        exit(0)
    train_fname = sys.argv[1]
    test_fname = sys.argv[2]
    feat_types_fname = sys.argv[3]
    cat_count_fname = sys.argv[4]
    work_dir = sys.argv[5]
    task = sys.argv[6]
    max_evals = int(sys.argv[7])
    n_cv_folds = int(sys.argv[8])
    num_trees = int(sys.argv[9])
    num_threads = int(sys.argv[10])
    time_log_fname = sys.argv[11]
    cat_type = sys.argv[12]
    if len(sys.argv) == 13:
        lgb_tuner = LightGBMTuner(train_fname, test_fname, feat_types_fname, cat_count_fname, work_dir,\
            task, max_evals, n_cv_folds, num_trees, num_threads, time_log_fname, cat_type)
    else:
        train_query_fname = sys.argv[13]
        test_query_fname = sys.argv[14]
        lgb_tuner = LightGBMTuner(train_fname, test_fname, feat_types_fname, cat_count_fname, work_dir, task,\
            max_evals, n_cv_folds, num_trees, num_threads, time_log_fname, cat_type, train_query_fname=train_query_fname, test_query_fname=test_query_fname)
    lgb_tuner.run()
