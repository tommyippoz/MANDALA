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
from mandalalib.classifiers.KerasClassifier import KerasClassifier
from mandalalib.classifiers.MANDALAClassifier import TabNet, FastAI, LogisticReg, UnsupervisedClassifier, XGB
from mandalalib.classifiers.PDIClassifier import PDIClassifier
from mandalalib.classifiers.PDITLClassifier import PDITLClassifier
from mandalalib.utils.MUtils import read_csv_dataset, read_csv_binary_dataset, get_clf_name, current_ms, \
    get_classifier_name

LABEL_NAME = 'label'
CSV_FOLDER = "split_binary_datasets"
OUTPUT_FOLDER = "./output_z/"

DIVERSITY_METRICS = [QStatMetric(), SigmaMetric(), CoupleDisagreementMetric(), DisagreementMetric(),
                     SharedFaultMetric()]


def get_ensembles(set1, set2, set3):
    e_list = []
    stackers = [DecisionTreeClassifier(), LinearDiscriminantAnalysis(),
                RandomForestClassifier(n_estimators=10), XGB(n_trees=10)]
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
            read_csv_dataset(os.path.join(CSV_FOLDER, dataset + "_train.csv"), label_name=LABEL_NAME,
                             split=False, shuffle=False)
        x_test, y_test, feature_list, att_perc = \
            read_csv_dataset(os.path.join(CSV_FOLDER, dataset + "_test.csv"), label_name=LABEL_NAME,
                             split=False, shuffle=False)

        for tr_c in x_train.columns:
            if tr_c not in x_test.columns:
                x_test[tr_c] = numpy.zeros(x_test.shape[0], dtype='int')

        for te_c in x_test.columns:
            if te_c not in x_train.columns:
                x_test = x_test.drop(columns=[te_c])

        algs = []
        tr_preds = []
        te_preds = []

        # Runs PDI Classifiers
        # for img_size in [40, 70]:
        #     for strat in ['pca', 'tsne']:
        #         clf = PDIClassifier(n_classes=len(numpy.unique(y_train)), img_size=img_size,
        #                             pdi_strategy=strat,
        #                             epochs=50, bsize=1024, val_split=0.3, verbose=0)
        #         clf.fit(x_train, y_train)
        #         algs.append(get_classifier_name(clf))
        #         tr_preds.append(clf.predict_proba(x_train))
        #         y_pred = clf.predict(x_test)
        #         te_preds.append(clf.predict_proba(x_test))
        #         print(get_clf_name(clf) + " MCC: " + str(sklearn.metrics.matthews_corrcoef(y_test, y_pred)))

        # Runs PDITL Classifiers
        for img_size in [40, 70]:
            for strat in ['pca', 'tsne']:
                clf = PDIClassifier(n_classes=len(numpy.unique(y_train)), img_size=img_size,
                                    pdi_strategy=strat,
                                    epochs=50, bsize=1024, val_split=0.3, verbose=0)
                clf.fit(x_train, y_train)
                algs.append(get_classifier_name(clf))
                tr_preds.append(clf.predict_proba(x_train))
                y_pred = clf.predict(x_test)
                te_preds.append(clf.predict_proba(x_test))
                print(get_clf_name(clf) + " MCC: " + str(sklearn.metrics.matthews_corrcoef(y_test, y_pred)))

                for clf_set in ['default', 'simple', 'mini']:
                    clf = PDITLClassifier(n_classes=len(numpy.unique(y_train)), img_size=img_size,
                                          pdi_strategy=strat, clf_net=clf_set,
                                          epochs=50, bsize=1024, val_split=0.3, verbose=0)
                    clf.fit(x_train, y_train)
                    algs.append(get_classifier_name(clf))
                    tr_preds.append(clf.predict_proba(x_train))
                    y_pred = clf.predict(x_test)
                    te_preds.append(clf.predict_proba(x_test))
                    print(get_clf_name(clf) + "[" + clf_set + "] MCC: " + str(
                        sklearn.metrics.matthews_corrcoef(y_test, y_pred)))

        # Runs Basic Classifiers
        # for clf in [KerasClassifier(n_features=x_train.shape[1], n_classes=len(numpy.unique(y_train))),
        #             GaussianNB(), LinearDiscriminantAnalysis(), DecisionTreeClassifier()]:
        #     clf.fit(x_train, y_train)
        #     algs.append(get_classifier_name(clf))
        #     tr_preds.append(clf.predict_proba(x_train))
        #     y_pred = clf.predict(x_test)
        #     te_preds.append(clf.predict_proba(x_test))
        #     print(get_clf_name(clf) + " MCC: " + str(sklearn.metrics.matthews_corrcoef(y_test, y_pred)))
        #
        # if not numpy.isnan(att_perc):
        #     cont = att_perc * 2 if att_perc * 2 < 0.5 else 0.5
        #     for clf in [UnsupervisedClassifier(COPOD(contamination=cont)),
        #                 UnsupervisedClassifier(IForest(contamination=cont, max_features=0.8, max_samples=0.8)),
        #                 UnsupervisedClassifier(HBOS(contamination=cont, n_bins=100)),
        #                 UnsupervisedClassifier(CBLOF(contamination=cont, alpha=0.75, beta=3))]:
        #         start_time = current_ms()
        #         clf.fit(x_train)
        #         algs.append(get_classifier_name(clf))
        #         tr_preds.append(clf.predict_proba(x_train))
        #         y_pred = clf.predict(x_test)
        #         te_preds.append(clf.predict_proba(x_test))
        #         print(get_clf_name(clf) + " MCC: " + str(
        #             sklearn.metrics.matthews_corrcoef(y_test, y_pred))
        #               + " Train time: " + str(current_ms() - start_time) + " ms")

        # Runs Tree-Based Classifiers
        # for clf in [XGB(), RandomForestClassifier()]:
        #     start_time = current_ms()
        #     clf.fit(x_train, y_train)
        #     algs.append(get_classifier_name(clf))
        #     tr_preds.append(clf.predict_proba(x_train))
        #     y_pred = clf.predict(x_test)
        #     te_preds.append(clf.predict_proba(x_test))
        #     print(get_clf_name(clf) + " Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred))
        #           + " Train time: " + str(current_ms() - start_time) + " ms")
        #
        # # Runs DL Tabular Classifiers
        # dl_clfs = []
        # for clf in [FastAI(), TabNet(epochs=40, verbose=1, patience=2)]:
        #     start_time = current_ms()
        #     clf.fit(x_train, y_train)
        #     algs.append(get_classifier_name(clf))
        #     tr_preds.append(clf.predict_proba(x_train))
        #     y_pred = clf.predict(x_test)
        #     te_preds.append(clf.predict_proba(x_test))
        #     print(get_clf_name(clf) + " Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred))
        #           + " Train time: " + str(current_ms() - start_time) + " ms")

        print_predictions(x_train, y_train, algs, tr_preds, dataset, "train")
        print_predictions(x_test, y_test, algs, te_preds, dataset, "test")
