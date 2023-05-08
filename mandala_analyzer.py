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
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRFClassifier

from mandalalib.EnsembleMetric import QStatMetric, SigmaMetric, CoupleDisagreementMetric, DisagreementMetric, \
    SharedFaultMetric
from mandalalib.MEnsemble import MEnsemble
from mandalalib.classifiers.MANDALAClassifier import TabNet, FastAI, LogisticReg, UnsupervisedClassifier, XGB
from mandalalib.utils.MUtils import read_csv_dataset, read_csv_binary_dataset, get_clf_name, current_ms, \
    get_classifier_name

OUTPUT_FOLDER = "./output/aggregateselectedrecall"

DIVERSITY_METRICS = [QStatMetric(), SigmaMetric(), CoupleDisagreementMetric(),
                     DisagreementMetric(relative=False),
                     SharedFaultMetric(relative=False), SharedFaultMetric(relative=True)]

STACKERS = [DecisionTreeClassifier(), LinearDiscriminantAnalysis(),
            RandomForestClassifier(n_estimators=10), XGB(n_trees=10)]

FOLDER_1 = "output/bindatasets_2selected"
FOLDER_2 = "output/bindatasets"


def merge_csvs(csv_1, csv_2):
    for col in csv_1.columns:
        if col not in csv_2.columns:
            csv_2[col] = csv_1[col]
    label_column = numpy.asarray(csv_2["label"])
    dataset = csv_2.drop(columns=["label"])

    for col in dataset.columns:
        if (col.startswith("GaussianNB")) or (col.startswith("LinearDisc")) or (col.startswith("TabNet(50")):
            dataset = dataset.drop(columns=[col])

    algs = {}
    for col in dataset.columns:
        if (not col.startswith("dataset_")) and (col.endswith("_pred")):
            algs[col.replace("_pred", "")] = numpy.asarray(dataset[col])
    return dataset, label_column, algs


if __name__ == '__main__':

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    available_datasets = []
    for file in os.listdir(FOLDER_1):
        if file.endswith(".csv"):
            available_datasets.append(file.replace("_train.csv", "").replace("_test.csv", ""))
    available_datasets = sorted(list(set(available_datasets)))

    base_metrics = {}
    similarity_metrics = {}
    clf_names = None

    for dataset_name in available_datasets:

        print("\n[DATASET] " + dataset_name)

        csv_1 = pandas.read_csv(os.path.join(FOLDER_1, dataset_name + "_test.csv"))
        csv_2 = pandas.read_csv(os.path.join(FOLDER_2, dataset_name + "_test.csv"))
        dataset_name = dataset_name.replace("_baselearners", "")

        dataset, label, clfs = merge_csvs(csv_1, csv_2)

        base_metrics[dataset_name] = {}
        for clf_name in clfs.keys():
            base_metrics[dataset_name][clf_name] = {}
            base_metrics[dataset_name][clf_name]["acc"] = sklearn.metrics.accuracy_score(clfs[clf_name], label)
            if base_metrics[dataset_name][clf_name]["acc"] < 0.5:
                clfs[clf_name] = 1 - clfs[clf_name]
                base_metrics[dataset_name][clf_name]["acc"] = sklearn.metrics.accuracy_score(clfs[clf_name], label)
            base_metrics[dataset_name][clf_name]["rec"] = sklearn.metrics.recall_score(clfs[clf_name], label)
            base_metrics[dataset_name][clf_name]["mcc"] = sklearn.metrics.matthews_corrcoef(clfs[clf_name], label)
            print("MCC of %.3f for classifier %s" % (base_metrics[dataset_name][clf_name]["mcc"], clf_name))

        if len(similarity_metrics.keys()) == 0:
            for dm in DIVERSITY_METRICS:
                similarity_metrics[dm.get_name()] = numpy.zeros((len(clfs), len(clfs)), dtype='float32')
        clf_names = list(clfs.keys())
        for i in tqdm(range(0, len(clf_names)), "clfs"):
            clf_1 = clfs[clf_names[i]]
            for j in range(i+1, len(clf_names)):
                clf_2 = clfs[clf_names[j]]
                couples = numpy.vstack([clf_1, clf_2]).T
                for dm in DIVERSITY_METRICS:
                    similarity_metrics[dm.get_name()][i][j] += dm.compute_diversity(couples, label)

    # Printing average distance metrics
    for dm in DIVERSITY_METRICS:
        similarity_metrics[dm.get_name()] /= len(available_datasets)
        df = pandas.DataFrame(data=similarity_metrics[dm.get_name()], columns=clf_names)
        df["clf_name"] = clf_names
        df.set_index('clf_name')
        df.to_csv(os.path.join(OUTPUT_FOLDER, dm.get_name() + ".csv"), index=True)

    # Printing Accuracies
    df = pandas.DataFrame(columns=clf_names)
    for dataset_name in base_metrics.keys():
        df1 = pandas.DataFrame(data=numpy.asarray([base_metrics[dataset_name][k]["acc"] for k in clf_names]).reshape(1, -1),
                               columns=clf_names, index=[dataset_name])
        df = df.append(df1)
    df.to_csv(os.path.join(OUTPUT_FOLDER, "accuracies.csv"), index=True)

    # Printing Recalls
    df = pandas.DataFrame(columns=clf_names)
    for dataset_name in base_metrics.keys():
        df1 = pandas.DataFrame(
            data=numpy.asarray([base_metrics[dataset_name][k]["rec"] for k in clf_names]).reshape(1, -1),
            columns=clf_names, index=[dataset_name])
        df = df.append(df1)
    df.to_csv(os.path.join(OUTPUT_FOLDER, "recalls.csv"), index=True)

    # Printing MCCs
    df = pandas.DataFrame(columns=clf_names)
    for dataset_name in base_metrics.keys():
        df1 = pandas.DataFrame(data=numpy.asarray([base_metrics[dataset_name][k]["mcc"] for k in clf_names]).reshape(1, -1),
                               columns=clf_names, index=[dataset_name])
        df = df.append(df1)
    df.to_csv(os.path.join(OUTPUT_FOLDER, "mccs.csv"), index=True)
