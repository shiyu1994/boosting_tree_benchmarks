import os
import sys

def gen(start_data_idx, end_data_idx):
    data_list = [
        ("higgs", "binary", "large"),
        ("yahoo", "ranking", "large"),
        ("msltr", "ranking", "large"),
        ("dataexpo_onehot", "binary", "large"),
        ("allstate", "binary", "small"),
        #("adult", "binary", "small"),
        #("amazon", "binary", "small"),
        #("appetency", "binary", "small"),
        #("click", "binary", "small"),
        #("internet", "binary", "small"),
        #("kick", "binary", "small"),
        #("upselling", "binary", "small"),
        #("nips_b", "binary", "large"),
        #("nips_c", "binary", "large"),
        ("year", "regression", "large")
    ]
    data_list = data_list[start_data_idx: end_data_idx]

    data_dir = "../data"
    accuracy_bash_file_name = "test_accuracy.sh"
    speed_bash_file_name = "test_speed.sh"
    memory_bash_file_name = "test_memory.sh"
    small_setting_leafwise = "params_leaf_wise_small.json"
    large_setting_leafwise = "params_leaf_wise_large.json"
    small_setting_symmetric = "params_symmetric_small.json"
    large_setting_symmetric = "params_symmetric_large.json"

    with open(accuracy_bash_file_name, "w") as accuracy_out_file,\
        open(speed_bash_file_name, "w") as speed_out_file,\
        open(memory_bash_file_name, "w") as memory_out_file:
        for data, obj, setting in data_list:
            data_name = data.split("/")[-1]
            if obj == "ranking":
                obj = "YetiRank"
                metrics = [("NDCG:top=1\;type=Exp", "1"), ("NDCG:top=3\;type=Exp", "3"), ("NDCG:top=5\;type=Exp", "5"), ("NDCG:top=10\;type=Exp", "10")]
            elif obj == "binary":
                obj = "Logloss"
                metrics = [("AUC", "")]
            elif obj == "regression":
                obj = "RMSE"
                metrics = [("RMSE", "")]
            data_path = data_dir + "/" + data
            memory_line_prefix = ("sleep 10s\n"
                "echo \"before run\"\n"
                "echo \"$(date '+%Y-%m-%d %H:%M:%S') $(free -m | grep Mem: | sed 's/Mem://g')\"\n"
                "echo \"before run\"\n")
            memory_line_suffix = ("sleep 50s\n"
                "echo \"$(date '+%Y-%m-%d %H:%M:%S') $(free -m | grep Mem: | sed 's/Mem://g')\"\n"
                "pkill catboost\n")
            if data == "yahoo" or data == "msltr":
                if setting == "large":
                    speed_line_leafwise = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set {0}.train.cat "
                        "--column-description {0}.train.cd --loss-function {1} 2>&1 | tee catboost_leafwise_{2}_speed.log\n").format(
                        data_path, obj, data_name, large_setting_leafwise
                    )
                    speed_line_symmetric = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set {0}.train.cat "
                        "--column-description {0}.train.cd --loss-function {1} 2>&1 | tee catboost_symmetric_{2}_speed.log\n").format(
                        data_path, obj, data_name, large_setting_symmetric
                    )
                    memory_line_leafwise = memory_line_prefix + ("catboost/catboost/app/catboost fit --params-file {3} --learn-set {0}.train.cat "
                        "--column-description {0}.train.cd --loss-function {1} 2>&1 | tee catboost_leafwise_{2}_speed.log &\n").format(
                        data_path, obj, data_name, large_setting_leafwise
                    ) + memory_line_suffix
                    memory_line_symmetric = memory_line_prefix + ("catboost/catboost/app/catboost fit --params-file {3} --learn-set {0}.train.cat "
                        "--column-description {0}.train.cd --loss-function {1} 2>&1 | tee catboost_symmetric_{2}_speed.log &\n").format(
                        data_path, obj, data_name, large_setting_symmetric
                    ) + memory_line_suffix
                else:
                    speed_line_leafwise = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set {0}.train.cat "
                        "--column-description {0}.train.cd --loss-function {1} 2>&1 | tee catboost_leafwise_{2}_speed.log &\n").format(
                        data_path, obj, data_name, small_setting_leafwise
                    )
                    speed_line_symmetric = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set {0}.train.cat "
                        "--column-description {0}.train.cd --loss-function {1} 2>&1 | tee catboost_symmetric_{2}_speed.log\n").format(
                        data_path, obj, data_name, small_setting_symmetric
                    )
                    memory_line_leafwise = memory_line_prefix + ("catboost/catboost/app/catboost fit --params-file {3} --learn-set {0}.train.cat "
                        "--column-description {0}.train.cd --loss-function {1} 2>&1 | tee catboost_leafwise_{2}_speed.log &\n").format(
                        data_path, obj, data_name, small_setting_leafwise
                    ) + memory_line_suffix
                    memory_line_symmetric = memory_line_prefix + ("catboost/catboost/app/catboost fit --params-file {3} --learn-set {0}.train.cat "
                        "--column-description {0}.train.cd --loss-function {1} 2>&1 | tee catboost_symmetric_{2}_speed.log &\n").format(
                        data_path, obj, data_name, small_setting_symmetric
                    ) + memory_line_suffix
                speed_out_file.write(speed_line_symmetric)
                speed_out_file.write(speed_line_leafwise)
                memory_out_file.write(memory_line_symmetric)
                memory_out_file.write(memory_line_leafwise)
                for metric, suffix in metrics:
                    if setting == "large":
                        accuracy_line_leafwise = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set {0}.train.cat --test-set {0}.test.cat "
                            "--column-description {0}.train.cd --loss-function {1} --eval-metric {4} 2>&1 | tee catboost_leafwise_{2}_accuracy_{5}.log\n").format(
                            data_path, obj, data_name, large_setting_leafwise, metric, suffix
                        )
                        accuracy_line_symmetric = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set {0}.train.cat --test-set {0}.test.cat "
                            "--column-description {0}.train.cd --loss-function {1} --eval-metric {4} 2>&1 | tee catboost_symmetric_{2}_accuracy_{5}.log\n").format(
                            data_path, obj, data_name, large_setting_symmetric, metric, suffix
                        )
                    else:
                        accuracy_line_leafwise = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set {0}.train.cat --test-set {0}.test.cat "
                            "--column-description {0}.train.cd --loss-function {1} --eval-metric {4} 2>&1 | tee catboost_leafwise_{2}_accuracy_{5}.log\n").format(
                            data_path, obj, data_name, small_setting_leafwise, metric, suffix
                        )
                        accuracy_line_symmetric = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set {0}.train.cat --test-set {0}.test.cat "
                            "--column-description {0}.train.cd --loss-function {1} --eval-metric {4} 2>&1 | tee catboost_symmetric_{2}_accuracy_{5}.log\n").format(
                            data_path, obj, data_name, small_setting_symmetric, metric, suffix
                        )
                    accuracy_out_file.write(accuracy_line_symmetric)
                    accuracy_out_file.write(accuracy_line_leafwise)
            else:
                if setting == "large":
                    speed_line_leafwise = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set libsvm://{0}.train "
                        "--column-description {0}.cd --loss-function {1} 2>&1 | tee catboost_leafwise_{2}_speed.log\n").format(
                        data_path, obj, data_name, large_setting_leafwise
                    )
                    speed_line_symmetric = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set libsvm://{0}.train "
                        "--column-description {0}.cd --loss-function {1} 2>&1 | tee catboost_symmetric_{2}_speed.log\n").format(
                        data_path, obj, data_name, large_setting_symmetric
                    )
                    memory_line_leafwise = memory_line_prefix + ("catboost/catboost/app/catboost fit --params-file {3} --learn-set libsvm://{0}.train "
                        "--column-description {0}.cd --loss-function {1} 2>&1 | tee catboost_leafwise_{2}_speed.log &\n").format(
                        data_path, obj, data_name, large_setting_leafwise
                    ) + memory_line_suffix
                    memory_line_symmetric = memory_line_prefix + ("catboost/catboost/app/catboost fit --params-file {3} --learn-set libsvm://{0}.train "
                        "--column-description {0}.cd --loss-function {1} 2>&1 | tee catboost_symmetric_{2}_speed.log &\n").format(
                        data_path, obj, data_name, large_setting_symmetric
                    ) + memory_line_suffix
                else:
                    speed_line_leafwise = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set libsvm://{0}.train "
                        "--column-description {0}.cd --loss-function {1} 2>&1 | tee catboost_leafwise_{2}_speed.log\n").format(
                        data_path, obj, data_name, small_setting_leafwise
                    )
                    speed_line_symmetric = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set libsvm://{0}.train "
                        "--column-description {0}.cd --loss-function {1} 2>&1 | tee catboost_symmetric_{2}_speed.log\n").format(
                        data_path, obj, data_name, small_setting_symmetric
                    )
                    memory_line_leafwise = memory_line_prefix + ("catboost/catboost/app/catboost fit --params-file {3} --learn-set libsvm://{0}.train "
                        "--column-description {0}.cd --loss-function {1} 2>&1 | tee catboost_leafwise_{2}_speed.log &\n").format(
                        data_path, obj, data_name, small_setting_leafwise
                    ) + memory_line_suffix
                    memory_line_symmetric = memory_line_prefix + ("catboost/catboost/app/catboost fit --params-file {3} --learn-set libsvm://{0}.train "
                        "--column-description {0}.cd --loss-function {1} 2>&1 | tee catboost_symmetric_{2}_speed.log &\n").format(
                        data_path, obj, data_name, small_setting_symmetric
                    ) + memory_line_suffix
                speed_out_file.write(speed_line_symmetric)
                speed_out_file.write(speed_line_leafwise)
                memory_out_file.write(memory_line_symmetric)
                memory_out_file.write(memory_line_leafwise)
                for metric, suffix in metrics:
                    if setting == "large":
                        accuracy_line_leafwise = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set libsvm://{0}.train --test-set libsvm://{0}.test "
                            "--column-description {0}.cd --loss-function {1} --eval-metric {4} 2>&1 | tee catboost_leafwise_{2}_accuracy_{5}.log\n").format(
                            data_path, obj, data_name, large_setting_leafwise, metric, suffix
                        )
                        accuracy_line_symmetric = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set libsvm://{0}.train --test-set libsvm://{0}.test "
                            "--column-description {0}.cd --loss-function {1} --eval-metric {4} 2>&1 | tee catboost_symmetric_{2}_accuracy_{5}.log\n").format(
                            data_path, obj, data_name, large_setting_symmetric, metric, suffix
                        )
                    else:
                        accuracy_line_leafwise = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set libsvm://{0}.train --test-set libsvm://{0}.test "
                            "--column-description {0}.cd --loss-function {1} --eval-metric {4} 2>&1 | tee catboost_leafwise_{2}_accuracy_{5}.log\n").format(
                            data_path, obj, data_name, small_setting_leafwise, metric, suffix
                        )
                        accuracy_line_symmetric = ("catboost/catboost/app/catboost fit --params-file {3} --learn-set libsvm://{0}.train --test-set libsvm://{0}.test "
                            "--column-description {0}.cd --loss-function {1} --eval-metric {4} 2>&1 | tee catboost_symmetric_{2}_accuracy_{5}.log\n").format(
                            data_path, obj, data_name, small_setting_symmetric, metric, suffix
                        )
                    accuracy_out_file.write(accuracy_line_symmetric)
                    accuracy_out_file.write(accuracy_line_leafwise)
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