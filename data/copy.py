import sys
import os

def copy(path):
    for fname in os.listdir(path):
        if fname.endswith(".type"):
            lines = []
            with open(path + "/" + fname, "r") as in_file:
                for line in in_file:
                    lines += [line]
            with open(path + "/" + fname, "w") as out_file:
                out_file.write("0:Numerical\n")
                for line in lines:
                    feat_id, feat_type = line.strip().split(":")
                    feat_id = int(feat_id) + 1
                    out_file.write("{0}:{1}\n".format(feat_id, feat_type))


if __name__ == "__main__":
    #copy_cd_type_count()
    copy("simplified_data/small")