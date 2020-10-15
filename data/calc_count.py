import numpy as np

data_list = ["adult", "amazon", "appetency", "click", "internet", "kick", "upselling", "nips_b", "nips_c"]
dir_path = "simplified_data"

def calc_count():
    for data in data_list:
        with open(dir_path + "/" + data + ".train", "r") as in_file,\
            open(dir_path + "/" + data + ".cd", "r") as type_file,\
            open(dir_path + "/" + data + ".count", "w") as count_file:
            categorical_feat_ids = []
            max_feat_id = 0
            for line in type_file:
                try:
                    feat_id, feat_type = line.strip().split("\t")
                except:
                    print(line, data)
                feat_id = int(feat_id)
                if feat_id > max_feat_id:
                    max_feat_id = feat_id
                if feat_type == "Categ":
                    categorical_feat_ids += [feat_id]
            if len(categorical_feat_ids) == 0:
                continue
            is_categorical = np.zeros(max_feat_id + 1, dtype=np.bool)
            feat_dict = {}
            for feat_id in categorical_feat_ids:
                is_categorical[feat_id] = True
                feat_dict[feat_id] = set()
            for line in in_file:
                elements = line.strip().split(" ")
                for element in elements[1:]:
                    feat_id, feat_val = element.split(":")
                    feat_id = int(feat_id)
                    if feat_id <= max_feat_id and is_categorical[feat_id]:
                        feat_dict[feat_id].add(feat_val)
            for feat_id in np.sort(categorical_feat_ids):
                count_file.write("{0}\t{1}\n".format(feat_id, len(feat_dict[feat_id])))

if __name__ == "__main__":
    calc_count()
            