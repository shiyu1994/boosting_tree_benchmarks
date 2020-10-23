import os
import sys

def gen(start_data_idx, end_data_idx):
    data_list = [
        ("small/higgs_small", "binary", ["dnn"], 28),
        ("small/yahoo_small", "ranking", ["dnn"], 700),
        ("small/msltr_small", "ranking", ["dnn"], 137),
        ("small/dataexpo_onehot_small", "binary", ["dnn"], 700),
        ("small/allstate_small", "binary", ["dnn"], 4228),
        ("adult", "binary", ["dnn", "deepfm"], 14),
        ("amazon", "binary", ["dnn", "deepfm"], 9),
        ("appetency", "binary", ["dnn", "deepfm"], 419),
        ("small/click_small", "binary", ["dnn", "deepfm"], 11),
        ("internet", "binary", ["dnn", "deepfm"], 68),
        ("kick", "binary", ["dnn", "deepfm"], 43),
        ("upselling", "binary", ["dnn", "deepfm"], 419),
        ("small/nips_b_small", "binary", ["dnn", "deepfm"], 25),
        ("small/nips_c_small", "binary", ["dnn", "deepfm"], 63),
        ("small/year_small", "regression", ["dnn"], 90)
    ]
    data_dir = "../data/"
    archs = ["10,10", "10,10,10", "50,50,25", "100,100,50", "100,100,100,50"]
    num_threads = 16
    batch_size = 100
    out_fname = "run.sh"
    with open(out_fname, "w") as out_file:
        for data, obj, modes, num_features in data_list:
            if obj != "binary":
                continue
            data_path = data_dir + data
            data_name = data.split("/")[-1]
            for mode in modes:
                for arch in archs:
                    arch_str = "_".join(arch.split(","))
                    line = "python -u {0}.py {1}.train.csv {1}.test.csv {1}.cd {1}.count {2} {3} {4} {5} > {0}_{6}_{7}.log\n".format(
                        mode,
                        data_path,
                        num_threads,
                        batch_size,
                        num_features,
                        arch,
                        data_name,
                        arch_str
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