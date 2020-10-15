data_list = ["appetency", "appetency_onehot", "upselling", "upselling_onehot"]
data_path = "simplified_data"

for data in data_list:
    for train_test in ["train", "test"]:
        lines = []
        with open(data_path + "/" + data + "." + train_test, "r") as in_file:
            for line in in_file:
                elements = line.strip().split(" ")
                if elements[0] == "-1":
                    elements[0] = "0"
                lines += " ".join(elements) + "\n"
        with open(data_path + "/" + data + "." + train_test, "w") as out_file:
            for line in lines:
                out_file.write(line)