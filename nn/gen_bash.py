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
    data_dir = "../data/"
    archs = ["20,20,10", "50,50,25", "100,100,50", "100,100,100,50"]
    num_threads = 16
    batch_size = 100
    num_epochs = 10
    out_fname = "run.sh"
    with open(out_fname, "w") as out_file:
        for data, obj, modes, num_features in data_list[start_data_id: end_data_idx]:
            if obj != "binary":
                continue
            data_path = data_dir + data
            data_name = data.split("/")[-1]
            for mode in modes:
                for arch in archs:
                    arch_str = "_".join(arch.split(","))
                    for seed in range(3):
                        line = "python -u {0}.py {1}.train.csv.norm.remap {1}.test.csv.norm.remap {1}.cd {1}.count {2} {3} {8} {4} {5} {0}_{6}_{7}_{9}.log {9}\n".format(
                            mode,
                            data_path,
                            num_threads,
                            batch_size,
                            num_features,
                            arch,
                            data_name,
                            arch_str,
                            num_epochs,
                            seed
                        )
                        out_file.write(line)
    os.system("chmod +x {0}".format(out_fname))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python gen_bash.py <start_data_id> <end_data_id>")
        exit(0)
    start_data_id = int(sys.argv[1])
    end_data_id = int(sys.argv[2])
    gen(start_data_id, end_data_id)
