import sys
import numpy as np

def sample(in_fname, out_fname, sample_count, query_fname=None, out_query_fname=None, query_sample_count=0):
    if query_fname is None:
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
    else:
        query_cnt = 0
        item_counts = []
        with open(query_fname, "r") as in_file:
            for line in in_file:
                query_cnt += 1
                item_counts += [int(line.strip())]
        query_indices = np.random.choice(query_cnt, query_sample_count, replace=False)
        query_indices = np.sort(query_indices)
        print("chose queries ", query_indices)
        query_indices = np.hstack((query_indices, np.array([-1])))
        cur_query_index = 0
        cur_chosen_query_pos = 0
        item_count = 0
        is_chosen = False
        with open(in_fname, "r") as in_file, open(out_fname, "w") as out_file,\
            open(out_query_fname, "w") as out_query_file:
            for line in in_file:
                item_count += 1
                if cur_query_index == query_indices[cur_chosen_query_pos]:
                    out_file.write(line)
                    is_chosen = True
                if item_count == item_counts[cur_query_index]:
                    if is_chosen:
                        is_chosen = False
                        cur_chosen_query_pos += 1
                        out_query_file.write("{0}\n".format(item_count))
                    item_count = 0
                    cur_query_index += 1

if __name__ == "__main__":
    if len(sys.argv) != 4 and len(sys.argv) != 7:
        print("usage: python sample.py <in_fname> <out_fname> <sample_count> [query_fname] [out_query_fname] [query_sample_count]")
        exit(0)
    in_fname = sys.argv[1]
    out_fname = sys.argv[2]
    sample_count = int(sys.argv[3])
    if len(sys.argv) == 4:
        sample(in_fname, out_fname, sample_count)
    else:
        query_fname = sys.argv[4]
        out_query_fname = sys.argv[5]
        query_sample_count = int(sys.argv[6])
        sample(in_fname, out_fname, sample_count, query_fname, out_query_fname, query_sample_count)