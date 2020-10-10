import sys
import numpy as np

def sample(in_fname, out_fname, sample_count):
    line_cnt = 0
    with open(in_fname, "r") as in_file:
        for line in in_file:
            line_cnt += 1
    indices = np.random.choice(line_cnt, sample_count, replace=False)
    indices = np.sort(indices)
    line_cnt = 0
    cur_index = 0
    with open(in_fname, "r") as in_file, open(out_fname, "w") as out_file:
        for line in in_file:
            if line_cnt == indices[cur_index]:
                out_file.write(line)
                cur_index += 1
                if cur_index == sample_count:
                    break
            line_cnt += 1

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: python sample.py <in_fname> <out_fname> <sample_count>")
        exit(0)
    in_fname = sys.argv[1]
    out_fname = sys.argv[2]
    sample_count = int(sys.argv[3])
    sample(in_fname, out_fname, sample_count)