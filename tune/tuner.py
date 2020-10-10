import hyperopt as hpt
import numpy as np
import time
import os

class Tuner:
    def __init__(self, train_fname, test_fname, feat_types_fname, work_dir, task, max_evals, n_cv_folds, time_log_fname):
        self.train_fname = train_fname
        self.test_fname = test_fname
        self.feat_types_fname = feat_types_fname
        self.work_dir = work_dir
        self.param_space = None
        self.task = task
        self.max_evals = max_evals
        self.n_cv_folds = n_cv_folds
        self.n_rseed = 5
        self.n_seed = 5
        self.time_log_fname = time_log_fname
        self.start_time = None
        self.test_time = 0.0
        self.time_tag = 0.0

        self.categorical_features = []
        if feat_types_fname.endswith(".type"):
            with open(self.feat_types_fname, "r") as in_file:
                for line in in_file:
                    fid, feat_type = line.strip().split(":")
                    fid = int(fid)
                    if feat_type == "Categorical":
                        self.categorical_features += [fid]
        elif feat_types_fname.endswith(".cd"):
            with open(self.feat_types_fname, "r") as in_file:
                offset = 0
                for line in in_file:
                    fid, feat_type = line.strip().split("\t")
                    fid = int(fid)
                    if feat_type == "Categ":
                        self.categorical_features += [fid - offset]
                    elif feat_type != "Num":
                        offset += 1
        else:
            raise NotImplementedError("unsupported feature type file {}".format(feat_types_fname))
            
        if not os.path.exists(work_dir):
            os.system("mkdir {}".format(work_dir))


    def eval(self, params, train_file, test_file, seed=0):
        raise NotImplementedError("method eval is not implemented in Tuner class")

    def fullfill_parameters(self):
         raise NotImplementedError("method fullfill_parameters is not implemented in Tuner class")

    def partition_train_data_for_cv(self, rseed):
        cv_train_fname_list = ["{0}/train{1}.txt".format(self.work_dir, i) for i in range(1, self.n_cv_folds + 1)]
        cv_test_fname_list = ["{0}/test{1}.txt".format(self.work_dir, i) for i in range(1, self.n_cv_folds + 1)]
        cv_test_files = [open(cv_test_fname, "w") for cv_test_fname in cv_test_fname_list]
        np.random.seed(rseed)
        with open(self.train_fname, "r") as in_file:
            for line in in_file:
                dest_file_id = np.random.randint(self.n_cv_folds)
                cv_test_files[dest_file_id].write(line)
        for i in range(self.n_cv_folds):
            dest_train_fname = cv_train_fname_list[i]
            source_fnames = []
            for index, test_fname in enumerate(cv_test_fname_list):
                if index != i:
                    source_fnames.append(test_fname)
            os.system("cat {0} > {1}".format(" ".join(source_fnames), dest_train_fname))
        return cv_train_fname_list, cv_test_fname_list

    def cv(self, params, train_files, test_files, seed):
        eval_scores = []
        status = hpt.STATUS_OK
        try:
            for train_file, test_file in zip(train_files, test_files):
                _, eval_score = self.eval(params, train_file, test_file, seed)
                eval_scores.append(eval_score)
        except Exception as err:
            print("failed at params " + str(params))
            status = hpt.STATUS_FAIL
            raise err
        if status == hpt.STATUS_OK:
            mean_eval_scores = np.mean(eval_scores, axis=0)
            var_eval_scores = np.var(eval_scores, axis=0)
            best_iter = np.argmax(mean_eval_scores) if self.task == "binary" or self.task == "ranking" else np.argmin(mean_eval_scores)
            best_score = mean_eval_scores[best_iter]
            best_loss = best_score if self.task == "regression" else 1 - best_score
            best_var = var_eval_scores[best_iter]
            cv_result = {
                "loss": best_loss,
                "var": best_var,
                "best_num_trees": 1 + best_iter,
                "params": params.copy(),
                "status": status,
                "best_iter": best_iter
            }
        else:
            cv_result = {
                "loss": np.nan,
                "status": status
            }
        self.log_into_file()
        self.time_tag = time.time() - self.start_time - self.test_time
        return cv_result

    def log_into_file(self):
        try:
            start_time = time.time()
            cur_best_params = self.trials.best_trial["result"]["params"]
            all_test_scores = []
            for seed in range(self.n_seed):
                _, test_score = self.eval(cur_best_params, self.train_fname, self.test_fname, seed=seed)
                all_test_scores.append(test_score)
            all_test_scores = np.array(all_test_scores)
            self.test_time += time.time() - start_time
            with open(self.time_log_fname, "a") as time_log_file:
                time_log_file.write("{0}:{1}\n".format(self.time_tag,\
                    ",".join(str(val) for val in all_test_scores[:, self.trials.best_trial["result"]["best_iter"]])))
        except:
            with open(self.time_log_fname, "a") as time_log_file:
                time_log_file.write("first round\n")

    def get_best_params_from_cv(self, rseed=0):
        self.trials = hpt.Trials()
        train_files, test_files = self.partition_train_data_for_cv(rseed)
        hpt.fmin(fn=lambda params: self.cv(params, train_files, test_files, rseed), space=self.param_space,
            algo=hpt.tpe.suggest, max_evals=self.max_evals, trials=self.trials, rstate=np.random.RandomState(rseed))
        self.log_into_file()
    
    def run(self):
        mean_test_scores = []
        var_test_scores = []
        all_test_scores = []
        for rseed in range(self.n_rseed):
            self.start_time = time.time()
            self.test_time = 0.0
            self.time_tag = 0.0
            with open(self.time_log_fname, "a") as time_log_file:
                time_log_file.write("rseed={}\n".format(rseed))
            self.get_best_params_from_cv(rseed=rseed)
            best_params = self.trials.best_trial["result"]["params"]
            print("best_params", best_params)
            best_iter = self.trials.best_trial["result"]["best_iter"]
            test_scores = []
            for seed in range(self.n_seed):
                _, test_score = self.eval(best_params, self.train_fname, self.test_fname, seed=seed)
                test_scores.append(test_score)
                all_test_scores.append(test_score[best_iter])
            mean_test_score = np.mean(test_scores, axis=0)
            var_test_score = np.var(test_scores, axis=0)
            print(test_scores[0][best_iter], test_scores[1][best_iter], test_scores[2][best_iter], test_scores[3][best_iter], test_scores[4][best_iter])
            print(var_test_score)
            print("test mean auc: %f, standard variance: %f, iteration: %d" % (mean_test_score[best_iter], np.sqrt(var_test_score[best_iter]), best_iter))
            print("param " + str(best_params))
            mean_test_scores += [mean_test_score[best_iter]]
            var_test_scores += [var_test_score[best_iter]]
        all_mean_by_group = np.mean(mean_test_scores)
        all_var_by_group = np.mean(var_test_scores)
        all_mean = np.mean(all_test_scores)
        all_var = np.var(all_test_scores)
        print("group test mean auc: %f, standard variance: %f" % (all_mean_by_group, np.sqrt(all_var_by_group)))
        print("all test mean auc: %f, standard variance: %f" % (all_mean, np.sqrt(all_var)))