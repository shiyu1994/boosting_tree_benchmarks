import os

data_list = [
    ("small/higgs_small", "binary"),
    ("small/yahoo_small", "ranking"),
    ("small/msltr_small", "ranking"),
    ("small/dataexpo_onehot_small", "binary"),
    ("small/allstate_small", "binary"),
    ("adult", "binary"),
    ("amazon", "binary"),
    ("appetency", "binary"),
    ("small/click_small", "binary"),
    ("internet", "binary"),
    ("kick", "binary"),
    ("upselling", "binary"),
    ("small/nips_b_small", "binary"),
    ("small/nips_c_small", "binary"),
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
        line = ("python -u ../lightgbm_tuner.py {0}.train {0}.test {0}.cd {0}.count tmp {1} {2} {3} {4} {5} {6}.log "
            "{0}.train.query {0}.test.query > {6}_tune.log\n").format(
            data_path, obj, n_trials, n_cv_folds, n_iterations, n_threads, data_name
        )
        lines += [line]
    else:
        line = ("python -u ../lightgbm_tuner.py {0}.train {0}.test {0}.cd {0}.count tmp {1} {2} {3} {4} {5} {6}.log > {6}_tune.log\n").format(
            data_path, obj, n_trials, n_cv_folds, n_iterations, n_threads, data_name
        )
        lines += [line]

with open(bash_file_name, "w") as out_file:
    for line in lines:
        out_file.write(line)
os.system("chmod +x {}".format(bash_file_name))