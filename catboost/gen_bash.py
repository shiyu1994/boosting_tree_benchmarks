import os
import sys

def gen(start_data_idx, end_data_idx):
    data_list = [
        ("higgs", "binary", "large"),
        ("yahoo", "ranking", "large"),
        ("msltr", "ranking", "large"),
        ("dataexpo_onehot", "binary", "large"),
        ("allstate", "binary", "small"),
        ("adult", "binary", "small"),
        ("amazon", "binary", "small"),
        ("appetency", "binary", "small"),
        ("click", "binary", "small"),
        ("internet", "binary", "small"),
        ("kick", "binary", "small"),
        ("upselling", "binary", "small"),
        ("nips_b", "binary", "large"),
        ("nips_c", "binary", "large"),
        ("year", "regression", "large")
    ]
    data_list = data_list[start_data_idx: end_data_idx]

    data_dir = "../data"
    accuracy_bash_file_name = "test_accuracy.sh"
    speed_bash_file_name = "test_speed.sh"
    small_setting_leafwise = "params_leaf_wise_small"
    large_setting_leafwise = "params_leaf_wise_large"
    small_setting_symmetric = "params_symmetric_small"
    large_setting_symmetric = "params_symmetric_large"

    with open(accuracy_bash_file_name, "w") as accuracy_out_file,\
        open(speed_bash_file_name, "w") as speed_out_file:
        for data, obj, setting in data_list:
            data_name = data.split("/")[-1]
            if obj == "ranking":
                obj = "YetiRank"
                metrics = ["NDCG@1", "NDCG@3", "NDCG@5", "NDCG@10"]
            elif obj == "binary":
                obj = "Logloss"
                metrics = ["AUC"]
            elif obj == "regression":
                obj = "RMSE"
                metrics = ["RMSE"]
            data_path = data_dir + "/" + data
            for metric in metrics:
                if setting == "large":
                    speed_line_leafwise = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set {0}.train "
                        "--column-description {0}.cd --loss-function {1} 2>&1 | tee xgboost_hist_{2}_speed.log\n").format(
                        data_path, obj, data_name, large_setting_leafwise
                    )
                    accuracy_line_leafwise = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set {0}.train --test-set {0}.test "
                        "--column-description {0}.cd --loss-function {1} --eval-metric {4} 2>&1 | tee xgboost_hist_{2}_accuracy.log\n").format(
                        data_path, obj, data_name, large_setting_leafwise, metric
                    )
                    speed_line_symmetric = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set {0}.train "
                        "--column-description {0}.cd --loss-function {1} 2>&1 | tee xgboost_hist_{2}_speed.log\n").format(
                        data_path, obj, data_name, large_setting_symmetric
                    )
                    accuracy_line_symmetric = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set {0}.train --test-set {0}.test "
                        "--column-description {0}.cd --loss-function {1} --eval-metric {4} 2>&1 | tee xgboost_hist_{2}_accuracy.log\n").format(
                        data_path, obj, data_name, large_setting_symmetric, metric
                    )
                else:
                    speed_line_leafwise = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set {0}.train "
                        "--column-description {0}.cd --loss-function {1} 2>&1 | tee xgboost_hist_{2}_speed.log\n").format(
                        data_path, obj, data_name, small_setting_leafwise
                    )
                    accuracy_line_leafwise = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set {0}.train --test-set {0}.test "
                        "--column-description {0}.cd --loss-function {1} --eval-metric {4} 2>&1 | tee xgboost_hist_{2}_accuracy.log\n").format(
                        data_path, obj, data_name, small_setting_leafwise, metric
                    )
                    speed_line_symmetric = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set {0}.train "
                        "--column-description {0}.cd --loss-function {1} 2>&1 | tee xgboost_hist_{2}_speed.log\n").format(
                        data_path, obj, data_name, small_setting_symmetric
                    )
                    accuracy_line_symmetric = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set {0}.train --test-set {0}.test "
                        "--column-description {0}.cd --loss-function {1} --eval-metric {4} 2>&1 | tee xgboost_hist_{2}_accuracy.log\n").format(
                        data_path, obj, data_name, small_setting_symmetric, metric
                    )
                speed_out_file.write(speed_line_leafwise)
                speed_out_file.write(speed_line_symmetric)
                accuracy_out_file.write(accuracy_line_leafwise)
                accuracy_out_file.write(accuracy_line_symmetric)
    os.system("chmod +x {}".format(speed_bash_file_name))
    os.system("chmod +x {}".format(accuracy_bash_file_name))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python gen_bash.py <start_data_id> <end_data_id>")
        exit(0)
    start_data_id = int(sys.argv[1])
    end_data_id = int(sys.argv[2])
    gen(start_data_id, end_data_id)