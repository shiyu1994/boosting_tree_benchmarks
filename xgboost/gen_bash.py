import os
import sys

def gen(start_data_idx, end_data_idx):
    data_list = [
        ("higgs", "binary", "large"),
        ("yahoo", "ranking", "large"),
        ("msltr", "ranking", "large"),
        ("dataexpo_onehot", "binary", "large"),
        ("allstate", "binary", "small"),
        #("adult_onehot", "binary", "small"),
        #("amazon_onehot", "binary", "small"),
        #("appetency_onehot", "binary", "small"),
        #("click_onehot", "binary", "small"),
        #("internet_onehot", "binary", "small"),
        #("kick_onehot", "binary", "small"),
        #("upselling_onehot", "binary", "small"),
        #("nips_b_onehot", "binary", "large"),
        #("nips_c_onehot", "binary", "large"),
        ("year", "regression", "large")
    ]
    small_setting_hist = "max_depth=0 max_leaves=127 eta=0.02 tree_method=hist grow_policy=lossguide"
    large_setting_hist = "max_depth=0 max_leaves=255 eta=0.1 tree_method=hist grow_policy=lossguide"
    small_setting_exact = "max_depth=7 eta=0.02 tree_method=exact"
    large_setting_exact = "tree_method=exact"
    data_list = data_list[start_data_idx: end_data_idx]

    data_dir = "../data"
    accuracy_bash_file_name = "test_accuracy.sh"
    speed_bash_file_name = "test_speed.sh"
    memory_bash_file_name = "test_memory.sh"

    memory_line_prefix = ("sleep 10s\n"
        "echo \"before run\"\n"
        "echo \"$(date '+%Y-%m-%d %H:%M:%S') $(free -m | grep Mem: | sed 's/Mem://g')\"\n"
        "echo \"before run\"\n")
    memory_line_suffix = ("sleep 50s\n"
        "echo \"$(date '+%Y-%m-%d %H:%M:%S') $(free -m | grep Mem: | sed 's/Mem://g')\"\n"
        "pkill xgboost\n")

    with open(accuracy_bash_file_name, "w") as accuracy_out_file,\
        open(speed_bash_file_name, "w") as speed_out_file,\
        open(memory_bash_file_name, "w") as memory_out_file:
        for data, obj, setting in data_list:
            data_path = data_dir + "/" + data
            data_name = data.split("/")[-1]
            if obj == "ranking":
                obj = "rank:pairwise"
                metric_is = "eval_metric=ndcg@1 eval_metric=ndcg@3 eval_metric=ndcg@5 eval_metric=ndcg@10"
            elif obj == "binary":
                obj = "binary:logistic"
                metric_is = "eval_metric=auc"
            elif obj == "regression":
                obj = "reg:linear"
                metric_is = "eval_metric=rmse"
            if setting == "large":
                speed_line_hist = ("xgboost/xgboost xgboost.conf data={0}.train objective={1} max_bin=255 "
                    "{3} 2>&1 | tee xgboost_hist_{2}_speed.log\n").format(
                    data_path, obj, data_name, large_setting_hist
                )
                memory_line_hist = memory_line_prefix + ("xgboost/xgboost xgboost.conf data={0}.train objective={1} max_bin=255 "
                    "{3} 2>&1 | tee xgboost_hist_{2}_speed.log &\n").format(
                    data_path, obj, data_name, large_setting_hist
                ) + memory_line_suffix
                accuracy_line_hist = ("xgboost/xgboost xgboost.conf data={0}.train eval[test]={0}.test objective={1} max_bin=255 "
                    "{3} {4} 2>&1 | tee xgboost_hist_{2}_accuracy.log\n").format(
                    data_path, obj, data_name, metric_is, large_setting_hist
                )
                speed_line_exact = ("xgboost/xgboost xgboost.conf data={0}.train objective={1} "
                    "{3} 2>&1 | tee xgboost_exact_{2}_speed.log\n").format(
                    data_path, obj, data_name, large_setting_exact
                )
                memory_line_exact = memory_line_prefix + ("xgboost/xgboost xgboost.conf data={0}.train objective={1} "
                    "{3} 2>&1 | tee xgboost_exact_{2}_speed.log &\n").format(
                    data_path, obj, data_name, large_setting_exact
                ) + memory_line_suffix
                accuracy_line_exact = ("xgboost/xgboost xgboost.conf data={0}.train eval[test]={0}.test objective={1} "
                    "{3} {4} 2>&1 | tee xgboost_exact_{2}_accuracy.log\n").format(
                    data_path, obj, data_name, metric_is, large_setting_exact
                )
            else:
                speed_line_hist = ("xgboost/xgboost xgboost.conf data={0}.train objective={1} max_bin=255 "
                    "{3} 2>&1 | tee xgboost_hist_{2}_speed.log\n").format(
                    data_path, obj, data_name, small_setting_hist
                )
                memory_line_hist = memory_line_prefix + ("xgboost/xgboost xgboost.conf data={0}.train objective={1} max_bin=255 "
                    "{3} 2>&1 | tee xgboost_hist_{2}_speed.log &\n").format(
                    data_path, obj, data_name, small_setting_hist
                ) + memory_line_suffix
                accuracy_line_hist = ("xgboost/xgboost xgboost.conf data={0}.train eval[test]={0}.test objective={1} max_bin=255 "
                    "{3} {4} 2>&1 | tee xgboost_hist_{2}_accuracy.log\n").format(
                    data_path, obj, data_name, metric_is, small_setting_hist
                )
                speed_line_exact = ("xgboost/xgboost xgboost.conf data={0}.train objective={1} "
                    "{3} 2>&1 | tee xgboost_exact_{2}_speed.log\n").format(
                    data_path, obj, data_name, small_setting_exact
                )
                memory_line_exact = memory_line_prefix + ("xgboost/xgboost xgboost.conf data={0}.train objective={1} "
                    "{3} 2>&1 | tee xgboost_exact_{2}_speed.log &\n").format(
                    data_path, obj, data_name, small_setting_exact
                ) + memory_line_suffix
                accuracy_line_exact = ("xgboost/xgboost xgboost.conf data={0}.train eval[test]={0}.test objective={1} "
                    "{3} {4} 2>&1 | tee xgboost_exact_{2}_accuracy.log\n").format(
                    data_path, obj, data_name, metric_is, small_setting_exact
                )
            speed_out_file.write(speed_line_exact)
            speed_out_file.write(speed_line_hist)
            accuracy_out_file.write(accuracy_line_exact)
            accuracy_out_file.write(accuracy_line_hist)
            memory_out_file.write(memory_line_exact)
            memory_out_file.write(memory_line_hist)
    os.system("chmod +x {}".format(speed_bash_file_name))
    os.system("chmod +x {}".format(accuracy_bash_file_name))
    os.system("chmod +x {}".format(memory_bash_file_name))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python gen_bash.py <start_data_id> <end_data_id>")
        exit(0)
    start_data_id = int(sys.argv[1])
    end_data_id = int(sys.argv[2])
    gen(start_data_id, end_data_id)