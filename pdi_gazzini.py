import copy
import os

import keras
import numpy
import pandas
import sklearn
import sklearn.model_selection
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix

from mandalalib.classifiers.PDIClassifier import PDIClassifier
from mandalalib.classifiers.PDITLClassifier import PDITLClassifier


class KerasClassifier:
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
                keras.metrics.Recall(name="accuracy"),
                keras.metrics.AUC(name="auc")
            ]
        )
        self.model = model

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
        pp = self.predict_proba(x_test)
        return numpy.argmax(pp, axis=1)

    def predict_proba(self, x_test):
        if isinstance(x_test, pandas.DataFrame):
            x_t = x_test.to_numpy()
        else:
            x_t = copy.deepcopy(x_test)
        x_t = self.normalize(x_t, compute_stats=False)
        return self.model.predict(x_t)

    def classifier_name(self):
        return "KerasCNN(" + str(self.epochs) + "-" + str(self.bsize) + "-" + str(self.val_split) + ")"


def read_csv_dataset(dataset_name, label_name="multilabel", limit=numpy.nan, split=True):
    """
    Method to process an input dataset as CSV
    :param normal_tag: tag that identifies normal data
    :param limit: integer to cut dataset if needed.
    :param dataset_name: name of the file (CSV) containing the dataset
    :param label_name: name of the feature containing the label
    :return: many values for analysis
    """
    # Loading Dataset
    df = pandas.read_csv(dataset_name, sep=",")

    # Shuffle
    df = df.sample(frac=1.0)
    df = df.fillna(0)
    df = df.replace('null', 0)
    df = df[df.columns[df.nunique() > 1]]

    # Testing Purposes
    if (numpy.isfinite(limit)) & (limit < len(df.index)):
        df = df[0:limit]

    if split:
        encoding = pandas.factorize(df[label_name])
        y_enc = encoding[0]
        labels = encoding[1]
    else:
        y_enc = df[label_name]

    # Basic Pre-Processing
    normal_frame = df.loc[df[label_name] == "normal"]
    print("\nDataset '" + dataset_name + "' loaded: " + str(len(df.index)) + " items, " + str(
        len(normal_frame.index)) + " normal and " + str(len(numpy.unique(df[label_name]))) + " labels")

    # Train/Test Split of Classifiers
    x = df.drop(columns=[label_name])
    x_no_cat = x.select_dtypes(exclude=['object'])
    feature_list = x_no_cat.columns

    if split:
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_no_cat, y_enc, test_size=0.5,
                                                                                    shuffle=True)
        return x_train, x_test, y_train, y_test, feature_list
    else:
        return x_no_cat, y_enc, feature_list


DATASETS_DIR = 'datasets_new'

if __name__ == '__main__':

    for file in os.listdir(DATASETS_DIR):
        if file.endswith(".csv"):
            print("Dataset : " + file)

            x_train, x_test, y_train, y_test, feature_list = read_csv_dataset(os.path.join(DATASETS_DIR, file),
                                                                              limit=10000)

            model = PDITLClassifier(n_classes=len(numpy.unique(y_train)), tl_tag='mnist',
                                    img_size=32, pdi_strategy='tsne',
                                  epochs=50, bsize=128, val_split=0.2, verbose=2)
            model.fit(x_train, y_train)
            y_test_pred = model.predict(x_test)

            acc = accuracy_score(y_true=y_test, y_pred=y_test_pred)
            mcc = matthews_corrcoef(y_true=y_test, y_pred=y_test_pred)
            matrix = confusion_matrix(y_true=y_test, y_pred=y_test_pred)
            print(f'Test Accuracy: {acc}')
            print('--------------------------')
            print(f'Matthews Correlation Coefficient: {mcc}')
            print('--------------------------')
            print(matrix)
