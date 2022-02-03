import keras_tuner as kt

from src.training import loss_functions, statistics
from src.training.loader import load_features
from src.training.network import generate_normalizer, generate_model, generate_triplet_model, data_generator
from src.training.statistics import generate_statistics

MAX_TRIALS = 100

EPOCHS_NUM = 5
EPOCHS_STEPS_NUM = 200

DENSE_LAYERS_MIN = 2
DENSE_LAYERS_MAX = 10
DENSE_LAYERS_STEP = 4

DENSE_LAYER_NODES_MIN = 32
DENSE_LAYER_NODES_MAX = 512
DENSE_LAYER_NODES_STEP = 32

LOSS_FUNCTION = loss_functions.triplet_loss_euler_2
OPTIMIZATION_STATISTIC = "balanced_accuracy"

SIMILARITY_THRESHOLDS = range(1, 75)
SIMILARITY_MEASURE = statistics.calc_manhattan_distance


class HyperRegressor(kt.HyperModel):
    def __init__(self, x_train, loss_function):
        self.x_train = x_train
        self.loss_function = loss_function
        super().__init__()

    def build(self, hp):
        nodes_per_layer = []
        for i in range(DENSE_LAYERS_MAX):
            nodes_per_layer.append(hp.Int(
                f"units_{i}",
                min_value=DENSE_LAYER_NODES_MIN,
                max_value=DENSE_LAYER_NODES_MAX,
                step=DENSE_LAYER_NODES_STEP
            ))

        dense_layers = []
        # Variable number of layers, to be optimized by the tuner
        for i in range(hp.Int(
                "layers",
                min_value=DENSE_LAYERS_MIN,
                max_value=DENSE_LAYERS_MAX,
                step=DENSE_LAYERS_STEP
        )):
            # Add a new configuration for a dense layer with a variable number of nodes, to be optimized by the tuner
            dense_layers.append(nodes_per_layer[i])

        normalizer = generate_normalizer(self.x_train)

        model = generate_model(normalizer, dense_layers)

        input_shape = (self.x_train.shape[1],)
        triplet_model = generate_triplet_model(model, input_shape)
        triplet_model.compile(
            optimizer="adam",
            loss=self.loss_function
        )

        return triplet_model

    def fit(self, hp, model, x, y, validation_data, **kwargs):
        model.fit(data_generator(x, y), **kwargs)
        x_val, y_val = validation_data

        embedding = model.layers[3]
        validation_output = embedding.predict(x_val)

        statistics = generate_statistics(SIMILARITY_THRESHOLDS, validation_output, y_val, SIMILARITY_MEASURE, True)

        best_threshold = None
        best_stats = None
        for threshold, stats in statistics:
            if best_stats is None or stats[OPTIMIZATION_STATISTIC] > best_stats[OPTIMIZATION_STATISTIC]:
                best_threshold = threshold
                best_stats = stats

        return {"threshold": best_threshold, **best_stats}


def tune(loss_function):
    data, (x_train, y_train), (x_val_1, y_val_1), (x_val_2, y_val_2), (x_test, y_test) = load_features()
    tuner = kt.RandomSearch(
        hypermodel=HyperRegressor(x_train, loss_function),
        max_trials=MAX_TRIALS,
        executions_per_trial=1,
        overwrite=True,
        directory="./output/network-tuning-balanced",
        project_name="triplet",
        objective=kt.Objective(OPTIMIZATION_STATISTIC, "max")
    )
    tuner.search_space_summary()
    tuner.search(x_train,
                 y_train,
                 epochs=EPOCHS_NUM,
                 steps_per_epoch=EPOCHS_STEPS_NUM,
                 validation_data=(x_val_1, y_val_1))
    models = tuner.get_best_models(num_models=5)
    best_model = models[0]
    best_model.summary()
    tuner.results_summary()


if __name__ == "__main__":
    tune(LOSS_FUNCTION)
