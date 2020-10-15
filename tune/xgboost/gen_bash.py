import os

data_list = [
    ("small/higgs_small", "binary"),
    ("small/yahoo_small", "ranking"),
    ("small/msltr_small", "ranking"),
    ("small/dataexpo_onehot_small", "binary"),
    ("small/allstate_small", "binary"),
    ("adult_onehot", "binary"),
    ("amazon_onehot", "binary"),
    ("appetency_onehot", "binary"),
    ("small/click_small_onehot", "binary"),
    ("internet_onehot", "binary"),
    ("kick_onehot", "binary"),
    ("upselling_onehot", "binary"),
    ("small/nips_b_small_onehot", "binary"),
    ("small/nips_c_small_onehot", "binary"),
    ("small/year_small", "regression")
]

data_dir = "../../data"
n_trials = 50
n_cv_folds = 4
n_iterations = 500
n_threads = 16
bash_file_name = "tune.sh"

lines = []
for data, obj in data_list:
    data_path = data_dir + "/" + data
    data_name = data.split("/")[-1]
    if obj == "ranking":
        for mode in ["exact", "hist"]:
            line = ("python -u ../xgboost_{7}_tuner.py {0}.train {0}.test {0}.cd tmp {1} {2} {3} {4} {5} {6}_{7}.log "
                "{0}.train.query {0}.test.query > {6}_{7}_tune.log\n").format(
                data_path, obj, n_trials, n_cv_folds, n_iterations, n_threads, data_name, mode
            )
            lines += [line]
    else:
        for mode in ["exact", "hist"]:
            line = ("python -u ../xgboost_{7}_tuner.py {0}.train {0}.test {0}.cd tmp {1} {2} {3} {4} {5} {6}_{7}.log > {6}_{7}_tune.log\n").format(
                data_path, obj, n_trials, n_cv_folds, n_iterations, n_threads, data_name, mode
            )
            lines += [line]

with open(bash_file_name, "w") as out_file:
    for line in lines:
        out_file.write(line)
os.system("chmod +x {}".format(bash_file_name))