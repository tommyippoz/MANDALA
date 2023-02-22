from math import log, e

import numpy
import pandas
import sklearn.metrics

import os

from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRFClassifier

from mandalalib.EnsembleMetric import QStatMetric, SigmaMetric, CoupleDisagreementMetric, DisagreementMetric, \
    SharedFaultMetric
from mandalalib.MEnsemble import MEnsemble
from mandalalib.classifiers.MANDALAClassifier import TabNet, FastAI, LogisticReg, UnsupervisedClassifier, XGB
from mandalalib.utils.MUtils import read_csv_dataset, read_csv_binary_dataset, get_clf_name, current_ms, \
    get_classifier_name

LABEL_NAME = 'label'
CSV_FOLDER = "split_binary_datasets"
OUTPUT_FOLDER = "./output/bindatasets"

DIVERSITY_METRICS = [QStatMetric(), SigmaMetric(), CoupleDisagreementMetric(), DisagreementMetric(),
                     SharedFaultMetric()]


def get_ensembles(set1, set2, set3, cont):
    e_list = []
    stackers = [DecisionTreeClassifier(), LinearDiscriminantAnalysis(),
                RandomForestClassifier(n_estimators=10), XGB(n_trees=10)]
    for ut in [False, True]:
        for adj in stackers:
            e_list.append(MEnsemble(models_folder="",
                                    classifiers=set1,
                                    diversity_metrics=DIVERSITY_METRICS,
                                    bin_adj=adj,
                                    use_training=ut))
    for ut in [False, True]:
        for adj in stackers:
            e_list.append(MEnsemble(models_folder="",
                                    classifiers=set2,
                                    diversity_metrics=DIVERSITY_METRICS,
                                    bin_adj=adj,
                                    use_training=ut))
    for ut in [False, True]:
        for adj in stackers:
            e_list.append(MEnsemble(models_folder="",
                                    classifiers=set3,
                                    diversity_metrics=DIVERSITY_METRICS,
                                    bin_adj=adj,
                                    use_training=ut))
    for c2 in set2:
        for c3 in set3:
            for ut in [False, True]:
                for adj in stackers:
                    e_list.append(MEnsemble(models_folder="",
                                            classifiers=[c2, c3],
                                            diversity_metrics=DIVERSITY_METRICS,
                                            bin_adj=adj,
                                            use_training=ut))
    for c1 in set1:
        for c2 in set2:
            for c3 in set3:
                for ut in [False, True]:
                    for adj in stackers:
                        e_list.append(MEnsemble(models_folder="",
                                                classifiers=[c1, c2, c3],
                                                diversity_metrics=DIVERSITY_METRICS,
                                                bin_adj=adj,
                                                use_training=ut))

    return e_list


def entropy(labels, base=None):
    """ Computes entropy of label distribution. """

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = numpy.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = numpy.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent


def print_predictions(dataset_x, dataset_y, algs, predictions, filename, tag):
    full_predictions = dataset_x
    full_predictions["label"] = dataset_y
    for i in range(0, len(algs)):
        preds = predictions[i]
        full_predictions[algs[i] + "_pred"] = numpy.argmax(preds, axis=1)
        full_predictions[algs[i] + "_maxprob"] = numpy.max(preds, axis=0)
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
            read_csv_dataset(os.path.join(CSV_FOLDER, dataset + "_train.csv"), label_name=LABEL_NAME, split=False,
                             limit=5000)
        x_test, y_test, feature_list, att_perc = \
            read_csv_dataset(os.path.join(CSV_FOLDER, dataset + "_test.csv"), label_name=LABEL_NAME, split=False,
                             limit=5000)

        algs = []
        tr_preds = []
        te_preds = []

        # Runs Basic Classifiers
        for clf in [GaussianNB(), LinearDiscriminantAnalysis(), DecisionTreeClassifier()]:
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

        print_predictions(x_train, y_train, algs, tr_preds, file, "train")
        print_predictions(x_test, y_test, algs, te_preds, file, "test")
