import os
from datetime import datetime
import json
import numpy as np
import sys
import os

nni_dir = os.path.expanduser("~/nni-experiments/")

def parse_datetime(line):
    dt = line.strip().split("[")[1].split("]")[0]
    date, time = dt.split(" ")
    year, month, day = date.split("-")
    hour, minute, second = time.split(":")
    year = int(year)
    month = int(month)
    day = int(day)
    hour = int(hour)
    minute = int(minute)
    second = int(second)
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)

def parse_params(line):
    param_str = line.strip().split('"parameters": ')[-1].split(', "parameter_index"')[0]
    return json.loads(param_str)

def parse_trial_id(line):
    id_str = line.strip().split("Trial job ")[-1].split(" status")[0]
    return id_str

def get_best_result(job_id, time_lim_in_hours, n_seed, min_max):
    time_lim_in_seconds = time_lim_in_hours * 3600
    dispatcher_path = nni_dir + job_id + "/log/dispatcher.log"
    nnimanager_path = nni_dir + job_id + "/log/nnimanager.log"
    mode, data_name = json.load(open(nni_dir + job_id + "/.config", "r"))["experimentConfig"]["experimentName"].split("_")
    with open(dispatcher_path, "r") as dispatcher_file:
        first_line = next(dispatcher_file)
        start_time = parse_datetime(first_line)
        
        best_loss_list = []
        for line in dispatcher_file:
            if line.find("trials with best loss") != -1:
                best_loss_list += [(parse_datetime(line), float(line.strip().split(" ")[-1]))]
    
    start_order = []
    param_list = []
    finish_order = []
    with open(nnimanager_path, "r") as nnimanager_file:
        for line in nnimanager_file:
            if line.find("NNIManager received command from dispatcher: TR,") != -1:
                params = parse_params(line)
                param_list += [params]
            if line.find("status changed from WAITING to RUNNING") != -1:
                trial_id = parse_trial_id(line)
                start_order += [trial_id]
            if line.find("status changed from RUNNING to SUCCEEDED") != -1:
                trial_id = parse_trial_id(line)
                finish_order += [trial_id]
        #print(len(start_order), len(param_list))
        #assert len(start_order) == len(param_list)
        param_dict = dict(zip(start_order, param_list))
    assert len(finish_order) == len(best_loss_list)
    best_i, best_loss, search_time_to_best = -1, np.inf, None
    for i, (dt, loss) in enumerate(best_loss_list):
        search_time_to_best = dt - start_time
        time = search_time_to_best.total_seconds()
        if time <= time_lim_in_seconds and loss < best_loss:
            best_i = i
            best_loss = loss
            search_time_to_best = dt - start_time
        elif time > time_lim_in_seconds:
            break
    best_trial_id = finish_order[best_i]
    best_param = param_dict[best_trial_id]
    learning_rate = 10 ** best_param["log_learning_rate"]

    if mode == "dnn":
        fname_prefix = "{3}_dnn_bs{0}_arch{1}_lr{2}".format(
            best_param["batch_size"],
            "_".join(best_param["layer_units"].split(",")),
            learning_rate,
            data_name
        )
    elif mode == "deepfm":
        fname_prefix = "{4}_deepfm_bs{0}_arch{1}_lr{2}_ed{3}".format(
            params["batch_size"],
            "_".join(params["layer_units"].split(",")),
            learning_rate,
            params["embed_dim"],
            data_name
        )
    all_res = []
    for seed in range(n_seed):
        fname = "{0}_seed{1}.log".format(fname_prefix, seed)
        res = np.genfromtxt(fname, delimiter=",", dtype=np.float, skip_header=1)[:, -1]
        if min_max == "max":
            res = np.max(res)
        else:
            res = np.min(res)
        all_res += [res]
    mean = np.mean(all_res)
    stdvar = np.std(all_res)
    print(mean, best_loss)
    if best_loss < 0.0:
        assert np.abs(mean - np.abs(best_loss)) <= 1e-6
    else:
        assert np.abs(mean - best_loss) <= 1e-6
    search_time_to_best_second = int(search_time_to_best.total_seconds())
    print("mean = %.6f, stdvar = %.6f, search time = %dh %dm %ds, search time in seconds = %ds" % (mean, stdvar,
        search_time_to_best_second // 3600, search_time_to_best_second % 3600 // 60, search_time_to_best_second % 60,
        search_time_to_best_second))
    

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("usage: python get_best_result.py <job_id> <time_lim_in_hours> <n_seed> <min_max>")
        exit(0)
    job_id = sys.argv[1]
    time_lim_in_hours = int(sys.argv[2])
    n_seed = int(sys.argv[3])
    min_max = sys.argv[4]
    get_best_result(job_id, time_lim_in_hours, n_seed, min_max)
