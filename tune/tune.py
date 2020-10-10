from lightgbm_tuner import LightGBMTuner
import sys

def tune(train_fname, test_fname, feat_types_fname, cat_count_fname, work_dir, task, max_evals, n_cv_folds, num_trees, num_threads, cat_converters):
    lgb_tuner = LightGBMTuner(train_fname, test_fname, feat_types_fname, cat_count_fname, work_dir, task, max_evals, n_cv_folds, num_trees, num_threads, cat_converters)
    lgb_tuner.run()

if __name__ == "__main__":
    if len(sys.argv) < 9:
        print("usage: python tune.py <data_dir> <work_dir> <task> <max_evals> <n_cv_folds> <num_trees> <num_threads> <cat_converters>")
        exit(0)
    data_dir = sys.argv[1]
    train_fname = data_dir + "/train.txt.encode"
    test_fname = data_dir + "/test.txt.encode"
    feat_types_fname = data_dir + "/feat_types.txt"
    cat_count_fname = data_dir + "/cat_count.txt"
    work_dir = sys.argv[2]
    task = sys.argv[3] 
    max_evals = int(sys.argv[4]) 
    n_cv_folds = int(sys.argv[5]) 
    num_trees = int(sys.argv[6])
    num_threads = int(sys.argv[7])
    cat_converters = sys.argv[8]
    tune(train_fname, test_fname, feat_types_fname, cat_count_fname, work_dir, task, max_evals, n_cv_folds, num_trees, num_threads, cat_converters)