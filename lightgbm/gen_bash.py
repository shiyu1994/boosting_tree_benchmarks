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
    small_setting = "num_leaves=127 learning_rate=0.02"
    data_list = data_list[start_data_idx: end_data_idx]

    data_dir = "../data"
    accuracy_bash_file_name = "test_accuracy.sh"
    speed_bash_file_name = "test_speed.sh"

    with open(accuracy_bash_file_name, "w") as accuracy_out_file,\
        open(speed_bash_file_name, "w") as speed_out_file:
        for data, obj, setting in data_list:
            data_name = data.split("/")[-1]
            if obj == "ranking":
                obj = "lambdarank"
                metric = "ndcg"
            elif obj == "binary":
                metric = "auc"
            elif obj == "regression":
                metric = "rmse"
            data_path = data_dir + "/" + data
            categorical_columns = []
            with open(data_path + ".cd", "r") as feature_type_file:
                for line in feature_type_file:
                    feat_id, feat_type = line.strip().split("\t")
                    feat_id = int(feat_id)
                    if feat_type == "Categ":
                        categorical_columns += [feat_id]
                categorical_columns = [str(col_id) for col_id in categorical_columns]
                if len(categorical_columns) > 0:
                    if setting == "large":
                        speed_line = ("LightGBM/lightgbm config=lightgbm.conf data={0}.train objective={1} "
                            "categorical_feature={3} 2>&1 | tee lightgbm_{2}_speed.log\n").format(
                            data_path, obj, data_name, ",".join(categorical_columns)
                        )
                        accuracy_line = ("LightGBM/lightgbm config=lightgbm.conf data={0}.train valid={0}.test objective={1} "
                            "metric={2} categorical_feature={4} 2>&1 | tee lightgbm_{3}_accuracy.log\n").format(
                            data_path, obj, metric, data_name, ",".join(categorical_columns)
                        )
                    else:
                        speed_line = ("LightGBM/lightgbm config=lightgbm.conf data={0}.train objective={1} "
                            "categorical_feature={3} {4} 2>&1 | tee lightgbm_{2}_speed.log\n").format(
                            data_path, obj, data_name, ",".join(categorical_columns), small_setting
                        )
                        accuracy_line = ("LightGBM/lightgbm config=lightgbm.conf data={0}.train valid={0}.test objective={1} "
                            "metric={2} categorical_feature={4} {5} 2>&1 | tee lightgbm_{3}_accuracy.log\n").format(
                            data_path, obj, metric, data_name, ",".join(categorical_columns), small_setting
                        )
                else:
                    if setting == "large":
                        speed_line = ("LightGBM/lightgbm config=lightgbm.conf data={0}.train objective={1} "
                            "2>&1 | tee lightgbm_{2}_speed.log\n").format(
                            data_path, obj, data_name
                        )
                        accuracy_line = ("LightGBM/lightgbm config=lightgbm.conf data={0}.train valid={0}.test objective={1} "
                            "metric={2} 2>&1 | tee lightgbm_{3}_accuracy.log\n").format(
                            data_path, obj, metric, data_name
                        )
                    else:
                        speed_line = ("LightGBM/lightgbm config=lightgbm.conf data={0}.train objective={1} "
                            "{3} 2>&1 | tee lightgbm_{2}_speed.log\n").format(
                            data_path, obj, data_name, small_setting
                        )
                        accuracy_line = ("LightGBM/lightgbm config=lightgbm.conf data={0}.train valid={0}.test objective={1} "
                            "metric={2} {4} 2>&1 | tee lightgbm_{3}_accuracy.log\n").format(
                            data_path, obj, metric, data_name, small_setting
                        )
                speed_out_file.write(speed_line)
                accuracy_out_file.write(accuracy_line)
        os.system("chmod +x {}".format(speed_bash_file_name))
        os.system("chmod +x {}".format(accuracy_bash_file_name))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python gen_bash.py <start_data_id> <end_data_id>")
        exit(0)
    start_data_id = int(sys.argv[1])
    end_data_id = int(sys.argv[2])
    gen(start_data_id, end_data_id)
