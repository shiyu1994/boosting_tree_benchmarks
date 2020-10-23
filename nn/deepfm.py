from dnn import *

class DeepFM(DNN):
    def __init__(self, train_fname, test_fname, feature_type_fname, cat_count_fname, num_threads, batch_size, num_features, layer_units):
        super().__init__(train_fname, test_fname, feature_type_fname, cat_count_fname, num_threads, batch_size, num_features, layer_units)

    def _model_func(self):
        inputs = {}
        for feature_index in range(self.num_features):
            inputs[str(feature_index)] = keras.Input(shape=(1,), name=str(feature_index))
        numeric_values = [inputs[str(feature_index)] for feature_index in self.numeric_feature]
        embeddings = {}
        for feature_index in self.cat_feature:
            embeddings[feature_index] = layers.Embedding(self.cat_count[feature_index] + 1, 5)(inputs[str(feature_index)])
        dense_input = layers.Concatenate()(numeric_values + list(embeddings.values()))
        fcn = layers.BatchNormalization()(dense_input)
        for unit in self.layer_units:
            fcn = layers.Dense(unit, activation="relu")(fcn)
        cat_feature_indices = np.sort(list(self.cat_feature))
        fm = 0
        for i in range(0, len(cat_feature_indices)):
            embedding_1 = embeddings[cat_feature_indices[i]]
            for j in range(i + 1, len(cat_feature_indices)):
                embedding_2 = embeddings[cat_feature_indices[j]]
            fm += tf.reduce_sum(embedding_1 * embedding_2)
        raw_score = layers.Dense(1, name="output")(fcn + fm)
        output = tf.sigmoid(raw_score)
        return keras.Model(inputs, output)


if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("usage: python deepfm.py <train_fname> <test_fname> <feature_type_fname> <cat_count_fname> <num_threads> <batch_size> <num_features> <layer_units>")
        exit(0)
    train_fname = sys.argv[1]
    test_fname = sys.argv[2]
    feature_type_fname = sys.argv[3]
    cat_count_fname = sys.argv[4]
    num_threads = int(sys.argv[5])
    batch_size = int(sys.argv[6])
    num_features = int(sys.argv[7])
    layer_units = sys.argv[8]
    deepfm = DeepFM(train_fname, test_fname, feature_type_fname, cat_count_fname, num_threads, batch_size, num_features, layer_units)
    deepfm.train()