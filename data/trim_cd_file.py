import re

data_list = ["year"]
data_dir = "simplified_data"

def trim():
    for data in data_list:
        lines = []
        with open(data_dir + "/" + data + ".cd", "r") as in_file:
            for line in in_file:
                elements = re.split(" |\t", line)
                non_empty = []
                for element in elements:
                    if element != "":
                        non_empty += [element]
                assert len(non_empty) == 2
                lines += ["\t".join(non_empty) + "\n"]
        with open(data_dir + "/" + data + ".cd", "w") as out_file:
            for line in lines:
                out_file.write(line)