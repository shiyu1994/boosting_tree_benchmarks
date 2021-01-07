import os
import sys

def gen(start_data_idx, end_data_idx):
    data_list = [
        ("higgs", "binary", ["dnn"], 28),
        ("yahoo", "ranking", ["dnn"], 700),
        ("msltr", "ranking", ["dnn"], 137),
        ("dataexpo_onehot", "binary", ["dnn"], 700),
        ("allstate", "binary", ["dnn"], 4228),
        ("adult", "binary", ["dnn", "deepfm"], 14),
        ("amazon", "binary", ["dnn", "deepfm"], 9),
        ("appetency", "binary", ["dnn", "deepfm"], 419),
        ("click", "binary", ["dnn", "deepfm"], 11),
        ("internet", "binary", ["dnn", "deepfm"], 68),
        ("kick", "binary", ["dnn", "deepfm"], 43),
        ("upselling", "binary", ["dnn", "deepfm"], 419),
        ("nips_b", "binary", ["dnn", "deepfm"], 25),
        ("nips_c", "binary", ["dnn", "deepfm"], 63),
        ("year", "regression", ["dnn"], 90)
    ]
    data_dir = "../data/nn/"
    out_fname = "run_nni.sh"
    with open(out_fname, "w") as out_file:
        yml_fname_list = []
        for data_name, obj, modes, num_features in data_list[start_data_id: end_data_idx]:
            data_path = data_dir + data_name
            optimize_mode = "maximize" if obj != "regression" else "minimize"
            for mode in modes:
                yml_fname = "{0}_{1}.yml".format(data_name, mode)
                command = "python {0}.py {1}.train.csv.norm.remap {1}.test.csv.norm.remap {1}.cd {1}.count {2} {3}".format(
                    mode, data_path, num_features, obj
                )
                with open(yml_fname, "w") as yml_file:
                    yml_file.write("authorName: Yu Shi\n"
                                   "experimentName: deepfm\n"
                                   "trialConcurrency: 8\n"
                                   "maxExecDuration: 24h\n"
                                   "maxTrialNum: 200\n"
                                   "trainingServicePlatform: local\n"
                                   "# The path to Search Space\n"
                                   "searchSpacePath: deepfm_params.json\n"
                                   "useAnnotation: false\n"
                                   "tuner:\n"
                                   "  builtinTunerName: TPE\n"
                                   "  classArgs:\n"
                                   "    optimize_mode: {1}\n"
                                   "# The path and the running command of trial\n"
                                   "trial:\n"
                                   "  command: {0}\n"
                                   "  codeDir: .\n"
                                   "  gpuNum: 0\n".format(command, optimize_mode))
                    out_file.write("nnictl create -f --config {0}\n".format(yml_fname))
    os.system("chmod +x {0}".format(out_fname))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python gen_bash.py <start_data_id> <end_data_id>")
        exit(0)
    start_data_id = int(sys.argv[1])
    end_data_id = int(sys.argv[2])
    gen(start_data_id, end_data_id)
