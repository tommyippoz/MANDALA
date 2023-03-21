import copy
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
    get_classifier_name, report, compute_feature_importances

LABEL_NAME = 'label'
CSV_FOLDER = "split_binary_datasets"
OUTPUT_FOLDER = "./output/"

DIVERSITY_METRICS = [QStatMetric(), SigmaMetric(), CoupleDisagreementMetric(), DisagreementMetric(),
                     SharedFaultMetric(relative=False), SharedFaultMetric(relative=True)]

STACKERS = [DecisionTreeClassifier(), LinearDiscriminantAnalysis(), GaussianNB(),
                RandomForestClassifier(n_estimators=10)]

FOLDER_1 = "output/bindatasets_2selected"
FOLDER_2 = "output/bindatasets"


def merge_csvs(csv_1, csv_2):
    for col in csv_1.columns:
        if col not in csv_2.columns:
            csv_2[col] = csv_1[col]
    label_column = numpy.asarray(csv_2["label"])
    dataset = csv_2.drop(columns=["label"])

    for col in dataset.columns:
        if (col.startswith("GaussianNB")) or (col.startswith("LinearDisc")) \
                or (col.startswith("TabNet(50")) or (col.startswith("TabNet(45")) \
                or (col.startswith("PDI(")):
            dataset = dataset.drop(columns=[col])

    algs = {}
    for col in dataset.columns:
        if (not col.startswith("dataset_")) and (col.endswith("_pred")):
            algs[col.replace("_pred", "")] = numpy.asarray(dataset[col])
    return dataset, label_column, algs


if __name__ == '__main__':

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    out_file = os.path.join(OUTPUT_FOLDER, "triple_ensemble.csv")

    with open(out_file, 'w') as f:
        f.write('dataset,adj,clfs,')
        for dm in DIVERSITY_METRICS:
            f.write(dm.get_name() + ",")
        f.write('matrix,acc,b_acc,mcc,logloss,best_base_acc,best_base_mcc,acc_gain,mcc_gain,\n')

    available_datasets = []
    for file in os.listdir(FOLDER_1):
        if file.endswith(".csv"):
            available_datasets.append(file.replace("_train.csv", "").replace("_test.csv", ""))
    available_datasets = sorted(list(set(available_datasets)))

    for dataset_name in available_datasets:

        print("\n[DATASET] " + dataset_name)

        csv_1 = pandas.read_csv(os.path.join(FOLDER_1, dataset_name + "_test.csv"))
        csv_2 = pandas.read_csv(os.path.join(FOLDER_2, dataset_name + "_test.csv"))
        dataset_name = dataset_name.replace("_baselearners", "")

        dataset, label, clfs = merge_csvs(csv_1, csv_2)
        x_train, x_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(dataset, label, test_size=0.5)
        original_dataset_columns = [x for x in dataset.columns if x.startswith("dataset_")]

        # Loops over MEnsembles
        for tree_alg in ["DecisionTree", "RandomForest", "XGB"]:

            tree_columns = [x for x in x_test.columns if x.startswith(tree_alg)]

            for dnn in ["FastAI", "TabNet(40"]:

                dnn_columns = [x for x in x_test.columns if x.startswith(dnn)]

                for pdi in ["PDITL_CNN(mobilenet-pca-40", "PDITL_CNN(mobilenet-pca-70", "KerasCNN"]:

                    pdi_columns = [x for x in x_test.columns if x.startswith(pdi)]

                    print("Using %s / %s / %s" % (tree_alg, dnn, pdi))

                    filtered_columns = original_dataset_columns + tree_columns + \
                                       dnn_columns + pdi_columns
                    clf_pred_columns = [x for x in filtered_columns if x.endswith("_pred")]

                    d_tr = x_train[filtered_columns]
                    d_te = x_test[filtered_columns]
                    clf_predictions = x_test[clf_pred_columns].to_numpy()

                    for stacker in STACKERS:

                        adj = copy.deepcopy(stacker)
                        # Exercises the Ensemble
                        start_time = current_ms()
                        adj.fit(d_tr, y_train)
                        ens_predictions = adj.predict(d_te)
                        end_time = current_ms()

                        # Reports MEnsemble Stats
                        metric_scores, clf_metrics = report(ens_predictions, clf_predictions,
                                                            y_test, DIVERSITY_METRICS, verbose=False)

                        print("Adjudicator " + str(get_classifier_name(adj))
                              + " MCC: " + str(clf_metrics["adj"]["mcc"])
                              + " Time: " + str(end_time - start_time) + " ms")

                        # Logging to file
                        with open(out_file, 'a') as f:
                            f.write(dataset_name + "," + get_classifier_name(adj) + "," +
                                    str(tree_alg) + "-" + str(dnn) + "-" + str(pdi) + ",")
                            for met in metric_scores.keys():
                                if isinstance(metric_scores[met], numpy.ndarray):
                                    f.write(numpy.array2string(metric_scores[met], separator=' ').replace('\n', '') + ",")
                                else:
                                    f.write(str(metric_scores[met]) + ",")
                            for met in clf_metrics["adj"].keys():
                                if isinstance(clf_metrics["adj"][met], numpy.ndarray):
                                    f.write(
                                        numpy.array2string(clf_metrics["adj"][met], separator=' ').replace('\n', '') + ",")
                                else:
                                    f.write(str(clf_metrics["adj"][met]) + ",")
                            f.write(",".join([str(x) for x in compute_feature_importances(adj)]))
                            f.write('\n')


