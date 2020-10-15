import sys

def get_cat_col(feat_type_fname):
    with open(feat_type_fname, "r") as in_file:
        cat_feat_list = []
        for line in in_file:
            feat_idx, feat_type = line.strip().split("\t")
            if feat_type == "Categ":
                cat_feat_list += [feat_idx]
        print(",".join(cat_feat_list))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python get_cat_col.py <feat_type_fname>")
        exit(0)
    feat_type_fname = sys.argv[1]
    get_cat_col(feat_type_fname)