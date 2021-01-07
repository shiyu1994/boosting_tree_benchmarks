from dnn import *

class DeepFM(DNN):
    def __init__(self, train_fname, test_fname, feature_type_fname, cat_count_fname,\
        num_threads, batch_size, num_epochs, num_features, layer_units, learning_rate, embed_dim,\
        objective, early_stopping_rounds, log_fname, seed):
        super().__init__(train_fname, test_fname, feature_type_fname, cat_count_fname,\
            num_threads, batch_size, num_epochs, num_features, layer_units, learning_rate, embed_dim, objective, early_stopping_rounds, log_fname, seed)

    def _model_func(self):
        inputs = {}
        for feature_index in range(self.num_features):
            inputs[str(feature_index)] = keras.Input(shape=(1,), name=str(feature_index))
        numeric_values = [inputs[str(feature_index)] for feature_index in self.numeric_feature]
        embeddings = {}
        for feature_index in self.cat_feature:
            embeddings[feature_index] = layers.Flatten()(layers.Embedding(self.cat_count[feature_index], self.embed_dim)(inputs[str(feature_index)]))
        dense_input = layers.Concatenate()(numeric_values + list(embeddings.values()))
        fcn = layers.BatchNormalization()(dense_input)
        for unit in self.layer_units:
            fcn = layers.Dense(unit, activation="relu")(fcn)
            fcn = layers.BatchNormalization()(fcn)
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

def run_trial(params, train_fname, test_fname, feature_type_fname, cat_count_fname, num_features, objective):
    all_seed_result = []
    data_name = train_fname.split("/")[-1].split(".")[0]
    learning_rate = 10 ** params["log_learning_rate"]
    for seed in range(params["n_seed"]):
        log_fname = "{5}_deepfm_bs{0}_arch{1}_lr{2}_ed{3}_seed{4}.log".format(
            params["batch_size"],
            "_".join(params["layer_units"].split(",")),
            learning_rate,
            params["embed_dim"],
            seed,
            data_name
        )
        model = DeepFM(
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
        print("usage: python deepfm.py <train_fname> <test_fname> <feature_type_fname>\
            <cat_count_fname> <num_threads> <batch_size> <num_epochs> <num_features> <layer_units> <log_fname> <seed>")
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
    deepfm = DeepFM(train_fname, test_fname, feature_type_fname, cat_count_fname, num_threads,\
        batch_size, num_epochs, num_features, layer_units, log_fname, seed)
    deepfm.train()
"""