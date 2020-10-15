import numpy as np

data_list = [("adult",15), ("amazon",10), ("appetency",420), ("click",12), ("internet",69), ("kick",44), ("upselling",420), ("nips_b",26), ("nips_c",64)]
data_dir = "simplified_data"

def onehot_encode(train_fname, test_fname, feat_type_fname,\
    out_train_fname, out_test_fname, out_type_fname, out_count_fname, num_feats):
    with open(feat_type_fname, "r") as feat_type_file, open(out_train_fname, "w") as out_train_file,\
        open(out_test_fname, "w") as out_test_file, open(out_type_fname, "w") as out_type_file,\
        open(out_count_fname, "w") as out_count_file:
        cat_dict = {}
        cat_code = {}
        for line in feat_type_file:
            feat_id, feat_type = line.strip().split("\t")
            feat_id = int(feat_id)
            if feat_type == "Categ":
                cat_dict[feat_id] = set()
                cat_code[feat_id] = {}
        is_categorical = np.zeros(num_feats, dtype=np.bool)
        for feat_id in cat_dict:
            is_categorical[feat_id] = True
        with open(train_fname, "r") as train_file, open(test_fname, "r") as test_file:
            def gather_cat_info(data_file):
                for line in data_file:
                    elements = line.strip().split(" ")
                    for element in elements[1:]:
                        feat_id, feat_val = element.split(":")
                        feat_id = int(feat_id)
                        if is_categorical[feat_id]:
                            if feat_val not in cat_dict[feat_id]:
                                cat_code[feat_id][feat_val] = len(cat_dict[feat_id])
                                cat_dict[feat_id].add(feat_val)
            gather_cat_info(train_file)
            gather_cat_info(test_file)
        offset = 0
        num_feat_map = {}
        for feat_id in range(num_feats):
            if not is_categorical[feat_id]:
                num_feat_map[feat_id] = offset
                offset += 1
        for feat_id in np.sort(list(cat_dict.keys())):
            for feat_val in cat_code[feat_id]:
                cat_code[feat_id][feat_val] += offset
            offset += len(cat_code[feat_id])
        with open(train_fname, "r") as train_file, open(test_fname, "r") as test_file:
            def encode_file(data_file, out_data_file):
                for line in data_file:
                    elements = line.strip().split(" ")
                    num_elements = []
                    cat_elements = []
                    for element in elements[1:]:
                        feat_id, feat_val = element.split(":")
                        feat_id = int(feat_id)
                        if is_categorical[feat_id]:
                            cat_elements += ["{0}:1".format(cat_code[feat_id][feat_val])]
                        else:
                            new_feat_id = num_feat_map[feat_id]
                            num_elements += ["{0}:{1}".format(new_feat_id, feat_val)]
                    new_line = " ".join([elements[0]] + num_elements + cat_elements) + "\n"
                    out_data_file.write(new_line)
            encode_file(train_file, out_train_file)
            encode_file(test_file, out_test_file)

def encode_all():
    for data, num_feats in data_list:
        train_fname = data_dir + "/" + data + ".train"
        test_fname = data_dir + "/" + data + ".test"
        type_fname = data_dir + "/" + data + ".cd"
        out_train_fname = data_dir + "/" + data + "_onehot.train"
        out_test_fname = data_dir + "/" + data + "_onehot.test"
        out_type_fname = data_dir + "/" + data + "_onehot.cd"
        out_count_fname = data_dir + "/" + data + "_onehot.count"
        try:
            onehot_encode(train_fname, test_fname, type_fname, out_train_fname, out_test_fname, out_type_fname, out_count_fname, num_feats)
        except Exception as err:
            print("failed for data {}".format(data))
            print("error message ", err)

if __name__ == "__main__":
    encode_all()