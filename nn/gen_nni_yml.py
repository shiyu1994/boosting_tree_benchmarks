import os
import sys

def gen(start_data_idx, end_data_idx):
    data_list = [
        ("higgs", "binary", ["dnn"], 28, "params_large_num"),
        ("yahoo", "regression", ["dnn"], 700, "params_large_num"),
        ("msltr", "regression", ["dnn"], 137, "params_large_num"),
        ("dataexpo", "binary", ["dnn", "deepfm"], 8, "params_large_cat"),
        ("allstate", "binary", ["dnn"], 30, "params_large_cat"),
        ("adult", "binary", ["dnn", "deepfm"], 14, "params_small_cat"),
        ("amazon", "binary", ["dnn", "deepfm"], 9, "params_small_cat"),
        ("appetency", "binary", ["dnn", "deepfm"], 419, "params_small_cat"),
        ("click", "binary", ["dnn", "deepfm"], 11, "params_small_cat"),
        ("internet", "binary", ["dnn", "deepfm"], 68, "params_small_cat"),
        ("kick", "binary", ["dnn", "deepfm"], 43, "params_small_cat"),
        ("upselling", "binary", ["dnn", "deepfm"], 419, "params_small_cat"),
        ("nips_b", "binary", ["dnn", "deepfm"], 25, "params_small_cat"),
        ("nips_c", "binary", ["dnn", "deepfm"], 49, "params_small_cat"),
        ("year", "regression", ["dnn"], 90, "params_large_num")
    ]
    data_dir = "../data/nn/"
    out_fname = "run_nni.sh"
    with open(out_fname, "w") as out_file:
        yml_fname_list = []
        for data_name, obj, modes, num_features, config_fname in data_list[start_data_id: end_data_idx]:
            data_path = data_dir + data_name
            optimize_mode = "maximize" if obj != "regression" else "minimize"
            for mode in modes:
                yml_fname = "{0}_{1}.yml".format(data_name, mode)
                command = "python {0}.py {1}.train.csv.norm.remap {1}.test.csv.norm.remap {1}.cd {1}.count {2} {3}".format(
                    mode, data_path, num_features, obj
                )
                with open(yml_fname, "w") as yml_file:
                    yml_file.write("authorName: Yu Shi\n"
                                   "experimentName: {3}_{4}\n"
                                   "trialConcurrency: 8\n"
                                   "maxExecDuration: 12h\n"
                                   "maxTrialNum: 200\n"
                                   "trainingServicePlatform: local\n"
                                   "# The path to Search Space\n"
                                   "searchSpacePath: {2}.json\n"
                                   "useAnnotation: false\n"
                                   "tuner:\n"
                                   "  builtinTunerName: TPE\n"
                                   "  classArgs:\n"
                                   "    optimize_mode: {1}\n"
                                   "# The path and the running command of trial\n"
                                   "trial:\n"
                                   "  command: {0}\n"
                                   "  codeDir: .\n"
                                   "  gpuNum: 0\n".format(command, optimize_mode, config_fname, mode, data_name))
                    out_file.write("nnictl create -f --config {0}\n".format(yml_fname))
    os.system("chmod +x {0}".format(out_fname))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python gen_bash.py <start_data_id> <end_data_id>")
        exit(0)
    start_data_id = int(sys.argv[1])
    end_data_id = int(sys.argv[2])
    gen(start_data_id, end_data_id)
