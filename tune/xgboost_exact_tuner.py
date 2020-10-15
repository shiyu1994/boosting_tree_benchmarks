from tuner import Tuner
import xgboost as xgb
import hyperopt as hpt
from hyperopt import hp
import numpy as np
import sys

class XGBoostTuner(Tuner):
    def __init__(self, train_fname, test_fname, feat_types_fname, work_dir, task, max_evals,\
            n_cv_folds, num_trees, num_threads, time_log_fname, train_query_fname=None, test_query_fname=None):
        super().__init__(train_fname, test_fname, feat_types_fname, work_dir, task, max_evals,\
            n_cv_folds, time_log_fname, train_query_fname=train_query_fname, test_query_fname=test_query_fname)
        print("using XGBoost in " + str(xgb.__file__))
        if self.task == "ranking":
            self.objective = "rank:ndcg"
        elif self.task == "binary":
            self.objective = "binary:logistic"
        elif self.task == "regression":
            self.objective = "reg:linear"
        else:
            raise NotImplementedError("Unknown task " + self.task)
        self.num_threads = num_threads
        self.num_trees = num_trees
        param_dict = {
            'eta': hp.loguniform('eta', -7, 0),
            'max_depth':hp.randint('max_depth', 9) + 2,
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
            'alpha': hp.choice('alpha', [0, hp.loguniform('alpha_log', -16, 2)]),
            'lambda': hp.choice('lambda', [0, hp.loguniform('lambda_log', -16, 2)]),
            'gamma': hp.choice('gamma', [0, hp.loguniform('gamma_log', -16, 2)]),
            "tree_method":"exact",
            "grow_policy":"depthwise"
        }
        self.param_space = param_dict
        self.metric = None


    def fullfill_parameters(self, params, seed):
        if self.objective == "binary:logistic":
            self.metric = "auc"
        elif self.objective == "reg:linear":
            self.metric = "rmse"
        elif self.objective == "rank:ndcg":
            self.metric = "ndcg@5"
        else:
            raise NotImplementedError("unknown objective type {}".format(self.objective))

        params.update({
            "objective": self.objective,
            "eval_metric": self.metric,
            "nthread": self.num_threads,
            "seed":seed
        })

    def eval(self, params, train_file, test_file, seed=0, train_query_fname=None, test_query_fname=None, early_stopping_rounds=None):
        train_data = xgb.DMatrix(train_file)
        test_data = xgb.DMatrix(test_file)
        if train_query_fname is not None:
            assert test_query_fname is not None
            train_group = np.genfromtxt(train_query_fname, delimiter=",", dtype=np.int)
            test_group = np.genfromtxt(test_query_fname, delimiter=",", dtype=np.int)
            train_data.set_group(train_group)
            test_data.set_group(test_group)
        eval_results = {}
        self.fullfill_parameters(params, seed)
        print("eval with params " + str(params))
        xgb_booster = xgb.train(params=params, dtrain=train_data, evals=[(test_data, "test")],\
            evals_result=eval_results, num_boost_round=self.num_trees, early_stopping_rounds=early_stopping_rounds)
        results = eval_results["test"][self.metric]
        train_data = None
        test_data = None
        return xgb_booster, results

if __name__ == "__main__":
    if len(sys.argv) != 11 and len(sys.argv) != 13:
        print("usage: python xgboost_exact_tuner.py <train_fname> <test_fname> <feat_types_fname>"
            "<work_dir> <task> <max_evals> <n_cv_folds> <num_trees> <num_threads> <time_log_fname> [train_query_fname test_query_fname]")
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
    if len(sys.argv) == 11:
        xgb_tuner = XGBoostTuner(train_fname, test_fname, feat_types_fname, work_dir, task, max_evals, n_cv_folds, num_trees, num_threads, time_log_fname)
    else:
        train_query_fname = sys.argv[11]
        test_query_fname = sys.argv[12]
        xgb_tuner = XGBoostTuner(train_fname, test_fname, feat_types_fname, work_dir, task, max_evals,\
            n_cv_folds, num_trees, num_threads, time_log_fname, train_query_fname=train_query_fname, test_query_fname=test_query_fname)
    xgb_tuner.run()