import sys
import numpy as np

def get(fname_prefix, n_seed, mode):
    all_res = []
    for seed in range(n_seed):
        fname = "{0}_seed{1}.log".format(fname_prefix, seed)
        res = np.genfromtxt(fname, delimiter=",", dtype=np.float, skip_header=1)[:, -1]
        if mode == "max":
            res = np.max(res)
        else:
            res = np.min(res)
        all_res += [res]
    print("mean: {0}, std: {1}".format(np.mean(all_res), np.std(all_res)))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: python get_mean_std_from_logs.py <fname_prefix> <n_seed> <mode>")
        exit(0)
    fname_prefix = sys.argv[1]
    n_seed = int(sys.argv[2])
    mode = sys.argv[3]
    get(fname_prefix, n_seed, mode)