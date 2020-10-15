import re

data_list = [
    ("small/higgs_small", "binary"),
    ("small/yahoo_small", "ranking"),
    ("small/msltr_small", "ranking"),
    ("small/dataexpo_onehot_small", "binary"),
    ("small/allstate_small", "binary"),
    ("adult", "binary"),
    ("amazon", "binary"),
    ("appetency", "binary"),
    ("small/click_small", "binary"),
    ("internet", "binary"),
    ("kick", "binary"),
    ("upselling", "binary"),
    ("small/nips_b_small", "binary"),
    ("small/nips_c_small", "binary"),
    ("small/year_small", "regression")
]
data_dir = "simplified_data"

def trim():
    for data, _ in data_list:
        lines = []
        with open(data_dir + "/" + data + ".cd", "r") as in_file:
            for line in in_file:
                elements = re.split(" |\t", line.strip())
                non_empty = []
                for element in elements:
                    if element != "":
                        non_empty += [element]
                assert len(non_empty) == 2
                lines += ["\t".join(non_empty) + "\n"]
        with open(data_dir + "/" + data + ".cd", "w") as out_file:
            for line in lines:
                out_file.write(line)

if __name__ == "__main__":
    trim()