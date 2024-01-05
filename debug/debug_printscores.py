import os

import numpy
import sklearn.metrics
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from mandalalib.EnsembleMetric import QStatMetric, SigmaMetric, CoupleDisagreementMetric, DisagreementMetric, \
    SharedFaultMetric
from mandalalib.classifiers.MANDALAClassifier import TabNet, FastAI, UnsupervisedClassifier, XGB
from mandalalib.classifiers.PDIClassifier import PDIClassifier
from mandalalib.classifiers.PDITLClassifier import PDITLClassifier
from mandalalib.utils.MUtils import read_csv_dataset, get_clf_name, current_ms, \
    get_classifier_name

LABEL_NAME = 'label'
CSV_FOLDER = "datasets"
OUTPUT_FOLDER = "./output/aaa"

DIVERSITY_METRICS = [QStatMetric(), SigmaMetric(), CoupleDisagreementMetric(), DisagreementMetric(),
                     SharedFaultMetric()]


def entropy(probs):
    norm_array = numpy.full(probs.shape[1], 1 / probs.shape[1])
    normalization = (-norm_array * numpy.log2(norm_array)).sum()
    ent = []
    for i in range(0, probs.shape[0]):
        val = numpy.delete(probs[i], numpy.where(probs[i] == 0))
        p = val / val.sum()
        ent.append(1 - (normalization - (-p * numpy.log2(p)).sum()) / normalization)
    return numpy.asarray(ent)


def print_predictions(dataset_x, dataset_y, algs, predictions, filename, tag):
    full_predictions = dataset_x
    full_predictions.columns = ["dataset_" + f for f in full_predictions.columns]
    full_predictions["label"] = dataset_y
    for i in range(0, len(algs)):
        preds = predictions[i]
        full_predictions[algs[i] + "_pred"] = numpy.argmax(preds, axis=1)
        full_predictions[algs[i] + "_maxprob"] = numpy.max(preds, axis=1)
        full_predictions[algs[i] + "_entropy"] = entropy(preds)

    full_predictions.to_csv(filename + "_baselearners_" + tag + ".csv", index=False)


if __name__ == '__main__':

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    available_datasets = []
    for file in os.listdir(CSV_FOLDER):
        if file.endswith(".csv"):
            available_datasets.append(file.replace("_train.csv", "").replace("_test.csv", ""))
    available_datasets = sorted(list(set(available_datasets)))

    for dataset in available_datasets:

        # Reads CSV Dataset
        x_train, y_train, feature_list, att_perc = \
            read_csv_dataset(os.path.join(CSV_FOLDER, dataset + "_train.csv"), label_name=LABEL_NAME, split=False)
        x_test, y_test, feature_list, att_perc = \
            read_csv_dataset(os.path.join(CSV_FOLDER, dataset + "_test.csv"), label_name=LABEL_NAME, split=False)

        algs = []
        tr_preds = []
        te_preds = []

        # Runs Basic Classifiers
        for clf in [PDIClassifier(), PDITLClassifier(n_classes=len()), DecisionTreeClassifier()]:
            clf.fit(x_train, y_train)
            algs.append(get_classifier_name(clf))
            tr_preds.append(clf.predict_proba(x_train))
            y_pred = clf.predict(x_test)
            te_preds.append(clf.predict_proba(x_test))
            print(get_clf_name(clf) + " Accuracy: " + str(sklearn.metrics.balanced_accuracy_score(y_test, y_pred)))

        if not numpy.isnan(att_perc):
            cont = att_perc * 2 if att_perc * 2 < 0.5 else 0.5
            for clf in [UnsupervisedClassifier(COPOD(contamination=cont)),
                        UnsupervisedClassifier(IForest(contamination=cont, max_features=0.8, max_samples=0.8)),
                        UnsupervisedClassifier(HBOS(contamination=cont, n_bins=100)),
                        UnsupervisedClassifier(CBLOF(contamination=cont, alpha=0.75, beta=3))]:
                start_time = current_ms()
                clf.fit(x_train)
                algs.append(get_classifier_name(clf))
                tr_preds.append(clf.predict_proba(x_train))
                y_pred = clf.predict(x_test)
                te_preds.append(clf.predict_proba(x_test))
                print(get_clf_name(clf) + " Accuracy: " + str(
                    sklearn.metrics.accuracy_score(y_test, y_pred))
                      + " Train time: " + str(current_ms() - start_time) + " ms")

        # Runs Tree-Based Classifiers
        for clf in [XGB(), RandomForestClassifier()]:
            start_time = current_ms()
            clf.fit(x_train, y_train)
            algs.append(get_classifier_name(clf))
            tr_preds.append(clf.predict_proba(x_train))
            y_pred = clf.predict(x_test)
            te_preds.append(clf.predict_proba(x_test))
            print(get_clf_name(clf) + " Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred))
                  + " Train time: " + str(current_ms() - start_time) + " ms")

        # Runs DL Tabular Classifiers
        dl_clfs = []
        for clf in [FastAI(), TabNet(epochs=40, verbose=1, patience=2)]:
            start_time = current_ms()
            clf.fit(x_train, y_train)
            algs.append(get_classifier_name(clf))
            tr_preds.append(clf.predict_proba(x_train))
            y_pred = clf.predict(x_test)
            te_preds.append(clf.predict_proba(x_test))
            print(get_clf_name(clf) + " Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred))
                  + " Train time: " + str(current_ms() - start_time) + " ms")

        print_predictions(x_train, y_train, algs, tr_preds, os.path.join(OUTPUT_FOLDER, dataset), "train")
        print_predictions(x_test, y_test, algs, te_preds, os.path.join(OUTPUT_FOLDER, dataset), "test")
