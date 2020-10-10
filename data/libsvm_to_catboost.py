import numpy as np
import sys

def libsvm_to_catboost(data_fname, feat_type_fname, out_data_fname, out_cd_fname, group_id_fname=None):
    with open(data_fname, "r") as data_file, open(feat_type_fname, "r") as feat_type_file,\
        open(out_data_fname, "w") as out_data_file, open(out_cd_fname, "w") as out_cd_file:
        if group_id_fname is not None:
            group_id_file = open(group_id_fname, "r")
        else:
            group_id_file = None
        num_features = 0
        offset = 0
        out_cd_file.write("0\tTarget\n")
        if group_id_file is not None:
            out_cd_file.write("1\tGroupId\n")
            offset = 1
        for line in feat_type_file:
            feat_id, feat_type = line.strip().split(":")
            feat_id = int(feat_id)
            assert feat_id == num_features
            num_features += 1
            if feat_type == "Categorical":
                feat_type = "Categ"
            elif feat_type == "Numerical":
                feat_type = "Num" 
            out_cd_file.write("{0}\t{1}\n".format(num_features + offset, feat_type))
        new_features = ["" for _ in range(num_features)]

        def with_group_id_file():
            group_counts = []
            for line in group_id_file:
                group_count = int(line.strip())
                group_counts += [group_count]
            cur_group = 0
            cur_cnt = 0
            total_cnt = 0
            line_cnt = 0
            for line in data_file:
                elements = line.strip().split(" ")
                label = elements[0]
                cur_cnt += 1
                line_cnt += 1
                for i in range(num_features):
                    new_features[i] = ""
                for element in elements[1:]:
                    feat_id, feat_val = element.split(":")
                    feat_id = int(feat_id)
                    new_features[feat_id] = feat_val
                new_line = "\t".join([label] + [str(cur_group)] + new_features) + "\n"
                out_data_file.write(new_line)
                if cur_cnt == group_counts[cur_group]:
                    cur_group += 1
                    total_cnt += cur_cnt
                    cur_cnt = 0
            assert total_cnt == line_cnt
        def without_group_id_file():
            for line in data_file:
                elements = line.strip().split(" ")
                label = elements[0]
                for i in range(num_features):
                    new_features[i] = ""
                for element in elements[1:]:
                    feat_id, feat_val = element.split(":")
                    feat_id = int(feat_id)
                    new_features[feat_id] = feat_val
                new_line = "\t".join([label] + new_features) + "\n"
                out_data_file.write(new_line)
        if group_id_file is None:
            without_group_id_file()
        else:
            with_group_id_file()
            group_id_file.close()

if __name__ == "__main__":
    if len(sys.argv) != 5 and len(sys.argv) != 6:
        print("usage: python libsvm_to_catboost.py <data_fname> <feat_type_fname> <out_data_fname> <out_cd_fname> [group_id_fname]")
        exit(0)
    data_fname = sys.argv[1]
    feat_type_fname = sys.argv[2]
    out_data_fname = sys.argv[3]
    out_cd_fname = sys.argv[4]
    if len(sys.argv) < 6:
        libsvm_to_catboost(data_fname, feat_type_fname, out_data_fname, out_cd_fname)
    else:
        group_id_fname = sys.argv[5]
        libsvm_to_catboost(data_fname, feat_type_fname, out_data_fname, out_cd_fname, group_id_fname)

            