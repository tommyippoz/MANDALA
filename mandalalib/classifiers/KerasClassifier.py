import copy

import keras
import numpy
import pandas
from keras.utils import to_categorical

from mandalalib.classifiers.MANDALAClassifier import MANDALAClassifier


class KerasClassifier(MANDALAClassifier):
    """
    Wrapper for a keras sequential network
    """

    def __init__(self, n_features, n_classes, epochs=50, bsize=1024, val_split=0.2, verbose=2):
        self.epochs = epochs
        self.bsize = bsize
        self.verbose = verbose
        self.val_split = val_split
        self.norm_stats = {"train_avg": None, "train_std": None}
        model = keras.Sequential(
            [
                keras.layers.Dense(
                    256, activation="relu", input_shape=(n_features,)
                ),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(n_classes, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer='adam', loss="binary_crossentropy", metrics=[
                keras.metrics.Accuracy(name="accuracy"),
                keras.metrics.AUC(name="auc")
            ]
        )
        MANDALAClassifier.__init__(self, model)

    def fit(self, x_train, y_train):
        self.classes_ = numpy.unique(y_train)
        if isinstance(x_train, pandas.DataFrame):
            x_t = x_train.to_numpy()
        else:
            x_t = copy.deepcopy(x_train)

        x_t = self.normalize(x_t, compute_stats=True)
        train_targets_cat = to_categorical(y_train, num_classes=len(self.classes_))
        self.model.fit(x_t, train_targets_cat,
                       batch_size=self.bsize,
                       epochs=self.epochs,
                       verbose=self.verbose,
                       validation_split=self.val_split)

        self.feature_importances_ = self.compute_feature_importances()
        self.trained = True

    def normalize(self, dataset, compute_stats=True):
        dataset = numpy.array(dataset, dtype='float32')
        if compute_stats or self.norm_stats["train_avg"] is None:
            self.norm_stats = {"train_avg": numpy.mean(dataset, axis=0),
                               "train_std": numpy.std(dataset, axis=0)}
        dataset -= self.norm_stats["train_avg"]
        dataset /= self.norm_stats["train_std"]
        return dataset

    def predict(self, x_test):
        return numpy.argmax(self.predict_proba(x_test), axis=1)

    def predict_proba(self, x_test):
        if isinstance(x_test, pandas.DataFrame):
            x_t = x_test.to_numpy()
        else:
            x_t = copy.deepcopy(x_test)
        x_t = self.normalize(x_t, compute_stats=False)
        return self.model.predict(x_t)

    def classifier_name(self):
        return "KerasCNN(" + str(self.epochs) + "-" + str(self.bsize) + "-" + str(self.val_split) + ")"
