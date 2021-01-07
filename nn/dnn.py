import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import sklearn
import sys
import nni

class DNN:
    def __init__(self, train_fname, test_fname, feature_type_fname, cat_count_fname,\
            num_threads, batch_size, num_epochs, num_features, layer_units, learning_rate, embed_dim, \
            objective, early_stopping_rounds, log_fname, seed):
        tf.random.set_seed(seed)
        self.train_fname = train_fname
        self.test_fname = test_fname
        self.cat_feature = set()
        self.numeric_feature = set()
        self.num_features = num_features
        with open(feature_type_fname, "r") as in_file:
            for line in in_file:
                feature_index, feature_type = line.strip().split("\t")
                feature_index = int(feature_index) - 1
                if feature_index < 0:
                    continue
                if feature_type == "Categ":
                    self.cat_feature.add(feature_index)
            for feature_index in range(self.num_features):
                if feature_index not in self.cat_feature:
                    self.numeric_feature.add(feature_index)
        self.cat_count = {}
        with open(cat_count_fname, "r") as in_file:
            for line in in_file:
                feature_index, cat_count = line.strip().split("\t")
                feature_index = int(feature_index) - 1
                if feature_index < 0:
                    continue
                self.cat_count[feature_index] = int(cat_count)
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.layer_units = list(map(lambda x: int(x), layer_units.split(",")))
        self.embed_dim = embed_dim
        self.model = self._model_func()
        self.log_fname = log_fname
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.objective = objective
        self.early_stopping_rounds = early_stopping_rounds

    def _model_func(self):
        inputs = {}
        for feature_index in range(self.num_features):
            inputs[str(feature_index)] = keras.Input(shape=(1,), name=str(feature_index))
        numeric_values = [inputs[str(feature_index)] for feature_index in self.numeric_feature]
        embeddings = [layers.Flatten()(layers.Embedding(self.cat_count[feature_index], self.embed_dim)(inputs[str(feature_index)]))\
            for feature_index in self.cat_feature]
        dense_input = layers.Concatenate()(numeric_values + embeddings)
        fcn = layers.BatchNormalization()(dense_input)
        for unit in self.layer_units:
            fcn = layers.Dense(unit, activation="relu")(fcn)
            fcn = layers.BatchNormalization()(fcn)
        raw_score = layers.Dense(1, name="output")(fcn)
        output = tf.sigmoid(raw_score)
        return keras.Model(inputs, output)

    def _input_func(self, fname, train_or_test):
        csv_data = tf.data.experimental.make_csv_dataset(
            fname,
            batch_size=self.batch_size,
            label_name="label",
            num_epochs=1
        )
        return csv_data

    def log_history(self, history):
        keys = []
        hists = []
        for key in history:
            keys += [key]
            hists += [history[key]]
        df = pd.DataFrame(data=np.array(hists).T, columns=keys)
        df.to_csv(self.log_fname)

    def train(self, seed):
        if self.objective == "binary":
            self.model.compile(optimizer=keras.optimizers.Adam(self.learning_rate),
                loss=keras.losses.BinaryCrossentropy(),
                metrics=[keras.metrics.AUC()])
            # should monitor early stopping with metric, but currently keras has a bug with monitoring with AUC
            # https://github.com/tensorflow/tensorflow/issues/43645
            metric = "auc"
            monitor_mode = "max"
            if seed > 0:
                metric += "_{0}".format(seed)
        elif self.objective == "regression":
            self.model.compile(optimizer=keras.optimizers.Adam(self.learning_rate),
                loss=keras.losses.MeanSquaredError(),
                metrics=[keras.metrics.RootMeanSquaredError()])
            metric = "loss"
            monitor_mode = "min"
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_{0}'.format(metric), mode=monitor_mode,
            patience=self.early_stopping_rounds)
        try:
            history_callback = self.model.fit(self._input_func(self.train_fname, "train"),
                batch_size=self.batch_size,
                epochs=self.num_epochs,
                validation_data=self._input_func(self.test_fname, "test"),
                verbose=True, callbacks=[callback])
            self.log_history(history_callback.history)
        except Exception as _:
            self.log_history(history_callback.history)


def run_trial(params, train_fname, test_fname, feature_type_fname, cat_count_fname, num_features, objective):
    all_seed_result = []
    data_name = train_fname.split("/")[-1].split(".")[0]
    learning_rate = 10 ** params["log_learning_rate"]
    for seed in range(params["n_seed"]):
        log_fname = "{4}_dnn_bs{0}_arch{1}_lr{2}_seed{3}.log".format(
            params["batch_size"],
            "_".join(params["layer_units"].split(",")),
            learning_rate,
            seed,
            data_name
        )
        model = DNN(
            train_fname,
            test_fname,
            feature_type_fname,
            cat_count_fname,
            params["num_threads"],
            params["batch_size"],
            params["num_epochs"],
            num_features,
            params["layer_units"],
            learning_rate,
            params["embed_dim"],
            objective,
            params["early_stopping_rounds"],
            log_fname,
            seed
        )
        model.train(seed)
        result = np.genfromtxt(log_fname, delimiter=",", dtype=np.float, skip_header=1)[:, -1]
        if objective == "binary":
            all_seed_result += [np.max(result)]
        elif objective == "regression":
            all_seed_result += [np.min(result)]
    all_seed_result = np.array(all_seed_result)
    mean = np.mean(all_seed_result)
    nni.report_final_result(mean)

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("usage: python deepfm.py <train_fname> <test_fname> <feature_type_fname> <cat_count_fname> <num_features> <objective>")
        exit(0)
    train_fname = sys.argv[1]
    test_fname = sys.argv[2]
    feature_type_fname = sys.argv[3]
    cat_count_fname = sys.argv[4]
    num_features = int(sys.argv[5])
    objective = sys.argv[6]
    params = nni.get_next_parameter()
    run_trial(params, train_fname, test_fname, feature_type_fname, cat_count_fname, num_features, objective)


"""
if __name__ == "__main__":
    if len(sys.argv) != 12:
        print("usage: python deepfm.py <train_fname> <test_fname> <feature_type_fname> <cat_count_fname>\
            <num_threads> <batch_size> <num_epochs> <num_features> <layer_units> <log_fname> <seed>")
        exit(0)
    train_fname = sys.argv[1]
    test_fname = sys.argv[2]
    feature_type_fname = sys.argv[3]
    cat_count_fname = sys.argv[4]
    num_threads = int(sys.argv[5])
    batch_size = int(sys.argv[6])
    num_epochs = int(sys.argv[7])
    num_features = int(sys.argv[8])
    layer_units = sys.argv[9]
    log_fname = sys.argv[10]
    seed = int(sys.argv[11])
    dnn = DNN(train_fname, test_fname, feature_type_fname, cat_count_fname,\
        num_threads, batch_size, num_epochs, num_features, layer_units, log_fname, seed)
    dnn.train()
"""