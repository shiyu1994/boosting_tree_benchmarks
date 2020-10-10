import sys

def libsvm_column_from_one(data_fname, out_data_fname, feat_type_fname=None, out_cd_fname=None):
    with open(data_fname, "r") as in_file, open(out_data_fname, "w") as out_file:
        for line in in_file:
            elements = line.strip().split(" ")
            label = elements[0]
            new_elements = []
            for element in elements[1:]:
                feat_idx, feat_val = element.split(":")
                feat_idx = int(feat_idx) + 1
                new_elements += ["{0}:{1}".format(feat_idx, feat_val)]
            new_line = " ".join([label] + new_elements) + "\n"
            out_file.write(new_line)
        if feat_type_fname is not None:
            assert out_cd_fname is not None
            with open(feat_type_fname, "r") as feat_type_file,\
                open(out_cd_fname, "w") as out_cd_file:
                out_cd_file.write("0\tTarget\n")
                for line in feat_type_file:
                    feat_id, feat_type = line.strip().split(":")
                    feat_id = int(feat_id) + 1
                    if feat_type == "Numerical":
                        out_cd_file.write("{0}\tNum\n".format(feat_id))
                    elif feat_type == "Categorical":
                        out_cd_file.write("{0}\tCateg\n".format(feat_id))

if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 5:
        print("usage: python libsvm_column_from_one.py <data_fname> <out_data_fname> [feat_type_fname out_cd_fname]")
        exit(0)
    data_fname = sys.argv[1]
    out_data_fname = sys.argv[2]
    if len(sys.argv) == 3:
        libsvm_column_from_one(data_fname, out_data_fname)
    else:
        feat_type_fname = sys.argv[3]
        out_cd_fname = sys.argv[4]
        libsvm_column_from_one(data_fname, out_data_fname, feat_type_fname, out_cd_fname)
