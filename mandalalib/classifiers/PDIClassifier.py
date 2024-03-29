import copy

import keras
import numpy
import pandas
import pyDeepInsight
from keras.utils import to_categorical

from mandalalib.classifiers.MANDALAClassifier import MANDALAClassifier


class PDIClassifier(MANDALAClassifier):
    """
    Wrapper for a keras sequential network
    """

    def __init__(self, n_classes, img_size=32, pdi_strategy='tsne', epochs=30, bsize=1024,
                 val_split=0.2, patience=10, verbose=2, model=None, is_rgb=False):
        self.img_t = None
        self.is_rgb = is_rgb
        self.img_size = img_size
        self.pdi_strategy = pdi_strategy
        self.epochs = epochs
        self.bsize = bsize
        self.verbose = verbose
        self.patience = patience
        self.val_split = val_split
        self.norm_stats = {"train_avg": None, "train_std": None}
        if model is None:
            model = keras.Sequential([
                    keras.layers.Conv2D(filters=8, kernel_size=3, padding='same',
                                        input_shape=(img_size, img_size, 1), activation='relu'),
                    keras.layers.BatchNormalization(),
                    keras.layers.MaxPooling2D(pool_size=2),
                    keras.layers.Flatten(),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(img_size, activation='relu'),
                    keras.layers.Dense(int(img_size / 2.0), activation='relu'),
                    keras.layers.Dense(n_classes, activation='softmax')
                ]
            )
            model.compile(
                optimizer='adam', loss="categorical_crossentropy", metrics=[
                    keras.metrics.CategoricalAccuracy(name='acc'),
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
        callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0,
                                                 patience=self.patience, verbose=0, mode="auto",
                                                 baseline=None, restore_best_weights=True)

        x_t = self.normalize_and_transform(x_t, compute_stats=True)
        train_targets_cat = to_categorical(y_train, num_classes=len(self.classes_))
        self.model.fit(x_t, train_targets_cat,
                       batch_size=self.bsize,
                       epochs=self.epochs,
                       verbose=self.verbose,
                       validation_split=self.val_split,
                       callbacks=[callback])

        self.feature_importances_ = self.compute_feature_importances()
        self.trained = True

    def normalize_and_transform(self, dataset, compute_stats=True):
        dataset = numpy.array(dataset, dtype='float32')
        if compute_stats or self.norm_stats["train_avg"] is None:
            self.norm_stats = {"train_avg": numpy.mean(dataset, axis=0),
                               "train_std": numpy.std(dataset, axis=0)}
            self.norm_stats["train_std"][self.norm_stats["train_std"] == 0.0] = 1.0
        dataset -= self.norm_stats["train_avg"]
        dataset /= self.norm_stats["train_std"]
        if self.img_t is None:
            if self.verbose > 0:
                print("Calling PyDeepInsight using strategy '" + self.pdi_strategy + "'")
            self.img_t = pyDeepInsight.ImageTransformer(pixels=(self.img_size, self.img_size),
                                                        feature_extractor=self.pdi_strategy)
            self.img_t.fit(dataset)
        if self.is_rgb:
            dataset = self.img_t.transform(dataset)
        else:
            dataset = self.img_t.transform(dataset, img_format='scalar')
        return dataset

    def predict(self, x_test):
        pp = self.predict_proba(x_test)
        return numpy.argmax(pp, axis=1)

    def predict_proba(self, x_test):
        if isinstance(x_test, pandas.DataFrame):
            x_t = x_test.to_numpy()
        else:
            x_t = copy.deepcopy(x_test)
        x_t = self.normalize_and_transform(x_t, compute_stats=False)
        return self.model.predict(x_t)

    def classifier_name(self):
        return "PDI(" + str(self.pdi_strategy) + "-" + str(self.img_size) + "-" + \
               str(self.epochs) + "-" + str(self.bsize) + "-" + str(self.val_split) + ")"
