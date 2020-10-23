import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import sklearn
import sys

class DNN:
    def __init__(self, train_fname, test_fname, feature_type_fname, cat_count_fname, num_threads, batch_size, num_features, layer_units, task, train_query_fname):
        self.train_fname = train_fname
        self.test_fname = test_fname
        self.cat_feature = set()
        self.numeric_feature = set()
        max_feature = 0
        with open(feature_type_fname, "r") as in_file:
            for line in in_file:
                feature_index, feature_type = line.strip().split("\t")
                feature_index = int(feature_index) - 1
                if feature_index < 0:
                    continue
                if feature_type == "Categ":
                    self.cat_feature.add(feature_index)
                else:
                    self.numeric_feature.add(feature_index)
                if feature_index > max_feature:
                    max_feature = feature_index
        assert max_feature == num_features - 1
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
        self.num_features = num_features
        self.layer_units = list(map(lambda x: int(x), layer_units.split(",")))
        self.model = self._model_func()

    def _model_func(self):
        inputs = {}
        for feature_index in range(self.num_features):
            inputs[str(feature_index)] = keras.Input(shape=(1,), name=str(feature_index))
        numeric_values = [inputs[str(feature_index)] for feature_index in self.numeric_feature]
        embeddings = [layers.Embedding(self.cat_count[feature_index] + 1, 5)(inputs[str(feature_index)])\
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
            num_epochs=2
        )
        return csv_data

    def train(self):
        self.model.compile(optimizer=keras.optimizers.RMSprop(),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[keras.metrics.AUC()])

        self.model.fit(self._input_func(self.train_fname, "train"),
            batch_size=self.batch_size,
            epochs=10,
            validation_data=self._input_func(self.test_fname, "test"))

if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("usage: python dnn.py <train_fname> <test_fname> <feature_type_fname> <cat_count_fname> <num_threads> <batch_size> <num_features> <layer_units>")
        exit(0)
    train_fname = sys.argv[1]
    test_fname = sys.argv[2]
    feature_type_fname = sys.argv[3]
    cat_count_fname = sys.argv[4]
    num_threads = int(sys.argv[5])
    batch_size = int(sys.argv[6])
    num_features = int(sys.argv[7])
    layer_units = sys.argv[8]
    dnn = DNN(train_fname, test_fname, feature_type_fname, cat_count_fname, num_threads, batch_size, num_features, layer_units)
    dnn.train()