import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import sklearn
import sys

class DNN:
    def __init__(self, train_fname, test_fname, feature_type_fname, cat_count_fname,\
        num_threads, batch_size, num_epochs, num_features, layer_units, log_fname, seed):
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
        self.model = self._model_func()
        self.log_fname = log_fname
        self.num_epochs = num_epochs

    def _model_func(self):
        inputs = {}
        for feature_index in range(self.num_features):
            inputs[str(feature_index)] = keras.Input(shape=(1,), name=str(feature_index))
        numeric_values = [inputs[str(feature_index)] for feature_index in self.numeric_feature]
        embeddings = [layers.Flatten()(layers.Embedding(self.cat_count[feature_index], 5)(inputs[str(feature_index)]))\
            for feature_index in self.cat_feature]
        dense_input = layers.Concatenate()(numeric_values + embeddings)
        fcn = layers.BatchNormalization()(dense_input)
        for unit in self.layer_units:
            fcn = layers.Dense(unit, activation="relu")(fcn)
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

    def train(self):
        self.model.compile(optimizer=keras.optimizers.RMSprop(),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[keras.metrics.AUC()])
        try:
            history_callback = self.model.fit(self._input_func(self.train_fname, "train"),
                batch_size=self.batch_size,
                epochs=self.num_epochs,
                validation_data=self._input_func(self.test_fname, "test"), verbose=True)
            self.log_history(history_callback.history)
        except Exception as _:
            self.log_history(history_callback.history)

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