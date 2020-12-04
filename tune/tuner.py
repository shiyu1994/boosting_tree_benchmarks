import hyperopt as hpt
import numpy as np
import time
import os

class TimeLimException(Exception):
    pass

class Tuner:
    def __init__(self, train_fname, test_fname, feat_types_fname, work_dir, task, max_evals, n_cv_folds,
        time_log_fname, train_query_fname=None, test_query_fname=None):
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
        self.time_lim = 17280
        self.early_stopping_rounds = 50
        self.is_first_log = True
        self.time_log_fname = time_log_fname
        self.start_time = None
        self.test_time = 0.0
        self.time_tag = 0.0
        self.categorical_features = []
        self.train_query_fname = train_query_fname
        self.test_query_fname = test_query_fname
        with open(self.feat_types_fname, "r") as in_file:
            for line in in_file:
                fid, feat_type = line.strip().split("\t")
                fid = int(fid)
                if feat_type == "Categ":
                    self.categorical_features += [fid]
        with open(self.time_log_fname, "w") as _:
            pass
            
        if not os.path.exists(work_dir):
            os.system("mkdir {}".format(work_dir))

    def eval(self, params, train_file, test_file, seed=0, train_query_fname=None, test_query_fname=None,\
        early_stopping_rounds=None, num_rounds=None):
        raise NotImplementedError("method eval is not implemented in Tuner class")

    def predict(self, params, test_file, seed, test_query_fname=None):
        raise NotImplementedError("method predict is not implemented in Tuner class")

    def fullfill_parameters(self):
         raise NotImplementedError("method fullfill_parameters is not implemented in Tuner class")

    def partition_train_data_for_cv(self, rseed):
        cv_train_fname_list = ["{0}/train{1}.txt".format(self.work_dir, i) for i in range(1, self.n_cv_folds + 1)]
        cv_test_fname_list = ["{0}/test{1}.txt".format(self.work_dir, i) for i in range(1, self.n_cv_folds + 1)]
        cv_test_files = [open(cv_test_fname, "w") for cv_test_fname in cv_test_fname_list]
        np.random.seed(rseed)
        if self.task == "ranking":
            cv_train_query_fname_list = ["{0}/train{1}.query".format(self.work_dir, i) for i in range(1, self.n_cv_folds + 1)]
            cv_test_query_fname_list = ["{0}/test{1}.query".format(self.work_dir, i) for i in range(1, self.n_cv_folds + 1)]
            cv_test_query_files = [open(cv_test_query_name, "w") for cv_test_query_name in cv_test_query_fname_list]
            fold_queries = [[] for _ in range(self.n_cv_folds)]
            item_counts = []
            with open(self.train_query_fname, "r") as train_query_file:
                cur_query_index = 0
                for line in train_query_file:
                    item_counts += [int(line.strip())]
                    dest_file_id = np.random.randint(self.n_cv_folds)
                    fold_queries[dest_file_id] += [cur_query_index]
                    cur_query_index += 1
            for fold_id in range(self.n_cv_folds):
                fold_queries[fold_id] += [-1]
            cur_chosen_query_pos = [0 for _ in range(self.n_cv_folds)]
            is_chosen_per_fold = [False for _ in range(self.n_cv_folds)]
            with open(self.train_fname, "r") as train_file:
                cur_query_index = 0
                cur_item_count = 0
                for line in train_file:
                    cur_item_count += 1
                    for fold_id in range(self.n_cv_folds):
                        if cur_query_index == fold_queries[fold_id][cur_chosen_query_pos[fold_id]]:
                            cv_test_files[fold_id].write(line)
                            is_chosen_per_fold[fold_id] = True
                    if cur_item_count == item_counts[cur_query_index]:
                        for fold_id in range(self.n_cv_folds):
                            if is_chosen_per_fold[fold_id]:
                                is_chosen_per_fold[fold_id] = False
                                cv_test_query_files[fold_id].write("{0}\n".format(cur_item_count))
                                cur_chosen_query_pos[fold_id] += 1
                        cur_item_count = 0
                        cur_query_index += 1
            for test_file, test_query_file in zip(cv_test_files, cv_test_query_files):
                test_file.close()
                test_query_file.close()
            for fold_id in range(self.n_cv_folds):
                with open(cv_test_fname_list[fold_id], "r") as test_file,\
                    open(cv_test_query_fname_list[fold_id], "r") as test_query_file:
                    line_count = 0
                    item_sum = 0
                    for line in test_file:
                        line_count += 1
                    for line in test_query_file:
                        item_sum += int(line.strip())
                    assert item_sum == line_count
            for i in range(self.n_cv_folds):
                dest_train_fname = cv_train_fname_list[i]
                dest_train_query_fname = cv_train_query_fname_list[i]
                source_fnames = []
                source_query_fnames = []
                for index, (test_fname, test_query_fname) in enumerate(zip(cv_test_fname_list, cv_test_query_fname_list)):
                    if index != i:
                        source_fnames.append(test_fname)
                        source_query_fnames.append(test_query_fname)
                os.system("cat {0} > {1}".format(" ".join(source_fnames), dest_train_fname))
                os.system("cat {0} > {1}".format(" ".join(source_query_fnames), dest_train_query_fname))
        else:
            cv_train_query_fname_list = [None for _ in range(self.n_cv_folds)]
            cv_test_query_fname_list = [None for _ in range(self.n_cv_folds)]
            with open(self.train_fname, "r") as in_file:
                for line in in_file:
                    dest_file_id = np.random.randint(self.n_cv_folds)
                    cv_test_files[dest_file_id].write(line)
                for test_file in cv_test_files:
                    test_file.close()
            for i in range(self.n_cv_folds):
                dest_train_fname = cv_train_fname_list[i]
                source_fnames = []
                for index, test_fname in enumerate(cv_test_fname_list):
                    if index != i:
                        source_fnames.append(test_fname)
                os.system("cat {0} > {1}".format(" ".join(source_fnames), dest_train_fname))
        return cv_train_fname_list, cv_test_fname_list, cv_train_query_fname_list, cv_test_query_fname_list

    def cv(self, params, train_files, test_files, seed, train_query_files=None, test_query_files=None):
        eval_scores = []
        status = hpt.STATUS_OK
        for train_file, test_file, train_query_file, test_query_file in zip(train_files, test_files, train_query_files, test_query_files):
            _, eval_score = self.eval(params, train_file, test_file, seed, train_query_file, test_query_file,\
                early_stopping_rounds=self.early_stopping_rounds)
            eval_scores.append(eval_score)
        assert len(eval_scores) == self.n_cv_folds
        if status == hpt.STATUS_OK:
            min_eval_len = len(eval_scores[0])
            for eval_score in eval_scores[1:]:
                if len(eval_score) < min_eval_len:
                    min_eval_len = len(eval_score)
            for i in range(len(eval_scores)):
                eval_scores[i] = eval_scores[i][:min_eval_len]
            if min_eval_len == 0:
                cv_result = {
                    "loss": np.nan,
                    "status": status
                }
            else:
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
        if not self.is_first_log:
            start_time = time.time()
            cur_best_params = self.trials.best_trial["result"]["params"]
            all_test_scores = []
            best_iter = self.trials.best_trial["result"]["best_iter"]
            best_num_rounds = self.trials.best_trial["result"]["best_num_trees"]
            assert best_num_rounds == best_iter + 1
            for seed in range(self.n_seed):
                _, test_score = self.eval(cur_best_params, self.train_fname, self.test_fname, seed=seed,\
                    train_query_fname=self.train_query_fname, test_query_fname=self.test_query_fname,\
                    num_rounds=best_num_rounds)
                assert best_iter < len(test_score)
                all_test_scores.append(test_score[best_iter])
            self.test_time += time.time() - start_time
            with open(self.time_log_fname, "a") as time_log_file:
                time_log_file.write("{0}:{1}\n".format(self.time_tag,\
                    ",".join(str(val) for val in all_test_scores)))
        else:
            self.is_first_log = False
        if self.time_tag >= self.time_lim:
            raise TimeLimException

    def get_best_params_from_cv(self, rseed=0):
        self.trials = hpt.Trials()
        train_files, test_files, train_query_files, test_query_files = self.partition_train_data_for_cv(rseed)
        hpt.fmin(fn=lambda params: self.cv(params, train_files, test_files, rseed, train_query_files, test_query_files), space=self.param_space,
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
            self.is_first_log = True
            with open(self.time_log_fname, "a") as time_log_file:
                time_log_file.write("rseed={}\n".format(rseed))
            try:
                self.get_best_params_from_cv(rseed=rseed)
            except TimeLimException:
                print("time limitation exceeds")
            best_params = self.trials.best_trial["result"]["params"]
            print("best_params", best_params)
            best_iter = self.trials.best_trial["result"]["best_iter"]
            best_num_rounds = self.trials.best_trial["result"]["best_num_trees"]
            assert best_num_rounds == best_iter + 1
            for seed in range(self.n_seed):
                _, test_score = self.eval(best_params, self.train_fname, self.test_fname, seed=seed,\
                    train_query_fname=self.train_query_fname, test_query_fname=self.test_query_fname,\
                    num_rounds=best_num_rounds)
                assert best_iter < len(test_score)
                all_test_scores.append(test_score[best_iter])
            mean_test_score = np.mean(all_test_scores)
            var_test_score = np.var(all_test_scores)
            print(var_test_score)
            print("test mean auc: %f, standard variance: %f, iteration: %d" % (mean_test_score, np.sqrt(var_test_score), best_iter))
            print("param " + str(best_params))
            mean_test_scores += [mean_test_score]
            var_test_scores += [var_test_score]
        all_mean_by_group = np.mean(mean_test_scores)
        all_var_by_group = np.mean(var_test_scores)
        all_mean = np.mean(all_test_scores)
        all_var = np.var(all_test_scores)
        print("group test mean auc: %f, standard variance: %f" % (all_mean_by_group, np.sqrt(all_var_by_group)))
        print("all test mean auc: %f, standard variance: %f" % (all_mean, np.sqrt(all_var)))