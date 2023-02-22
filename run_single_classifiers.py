import numpy
import sklearn.metrics

import os

from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from mandalalib.EnsembleMetric import QStatMetric, SigmaMetric, CoupleDisagreementMetric, DisagreementMetric, \
    SharedFaultMetric
from mandalalib.classifiers.MANDALAClassifier import TabNet, FastAI, UnsupervisedClassifier, XGB
from mandalalib.utils.MUtils import read_csv_binary_dataset, get_clf_name, current_ms, read_csv_dataset

LABEL_NAME = 'multilabel'
CSV_FOLDER = "datasets"
OUTPUT_FILE = "./output/single_multilabel_scores.csv"

DIVERSITY_METRICS = [QStatMetric(), SigmaMetric(), CoupleDisagreementMetric(), DisagreementMetric(),
                     SharedFaultMetric()]

if __name__ == '__main__':

    with open(OUTPUT_FILE, 'w') as f:
        f.write('dataset,clf,time,matrix,acc,b_acc,mcc,logloss\n')

    for file in os.listdir(CSV_FOLDER):
        if file.endswith(".csv"):

            # Reads CSV Dataset
            x_train, x_test, y_train, y_test, feature_list, att_perc = \
                read_csv_dataset(os.path.join(CSV_FOLDER, file))

            # u_clfs = []
            # cont = att_perc*2 if att_perc*2 < 0.5 else 0.5
            # for clf in [UnsupervisedClassifier(COPOD(contamination=cont)),
            #             UnsupervisedClassifier(IForest(contamination=cont, max_features=0.8, max_samples=0.8)),
            #             UnsupervisedClassifier(HBOS(contamination=cont, n_bins=100)),
            #             UnsupervisedClassifier(CBLOF(contamination=cont, alpha=0.75, beta=3))]:
            #     start_time = current_ms()
            #     clf.fit(x_train, y_train)
            #     u_clfs.append(clf)
            #     y_pred = clf.predict(x_test)
            #     print(get_clf_name(clf) + " Accuracy: " + str(
            #         sklearn.metrics.accuracy_score(y_test, y_pred))
            #           + " Train time: " + str(current_ms() - start_time) + " ms")
            #
            #     # Logging to file
            #     with open(OUTPUT_FILE, 'a') as f:
            #         f.write(file + "," + get_clf_name(clf) + "," + str(current_ms() - start_time) + ","
            #                 + str(sklearn.metrics.confusion_matrix(y_test, y_pred).flatten()) + ","
            #                 + str(sklearn.metrics.accuracy_score(y_test, y_pred)) + ","
            #                 + str(sklearn.metrics.balanced_accuracy_score(y_test, y_pred)) + ","
            #                 + str(sklearn.metrics.matthews_corrcoef(y_test, y_pred)) + ","
            #                 + str(sklearn.metrics.log_loss(y_test, y_pred)) + "\n")

            # Runs Tree-Based Classifiers
            tb_clfs = []
            for clf in [XGB(), RandomForestClassifier(), DecisionTreeClassifier()]:
                start_time = current_ms()
                clf.fit(x_train, y_train)
                tb_clfs.append(clf)
                y_proba = clf.predict_proba(x_test)
                y_pred = clf.predict(x_test)
                print(get_clf_name(clf) + " Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred))
                      + " Train time: " + str(current_ms() - start_time) + " ms")

                # Logging to file
                with open(OUTPUT_FILE, 'a') as f:
                    f.write(file + "," + get_clf_name(clf) + "," + str(current_ms() - start_time) + ","
                            + str(sklearn.metrics.confusion_matrix(y_test, y_pred).flatten()) + ","
                            + str(sklearn.metrics.accuracy_score(y_test, y_pred)) + ","
                            + str(sklearn.metrics.balanced_accuracy_score(y_test, y_pred)) + ","
                            + str(sklearn.metrics.matthews_corrcoef(y_test, y_pred)) + ","
                            + str(sklearn.metrics.log_loss(y_test, y_proba)) + "\n")

            # Runs DL Tabular Classifiers
            dl_clfs = []
            for clf in [FastAI(feature_names=feature_list), TabNet(epochs=40, verbose=1, patience=2), TabNet(epochs=100, verbose=1, patience=2)]:
                start_time = current_ms()
                clf.fit(x_train, y_train)
                dl_clfs.append(clf)
                y_proba = clf.predict_proba(x_test)
                y_pred = clf.predict(x_test)
                print(get_clf_name(clf) + " Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred))
                      + " Train time: " + str(current_ms() - start_time) + " ms")

                # Logging to file
                with open(OUTPUT_FILE, 'a') as f:
                    f.write(file + "," + get_clf_name(clf) + "," + str(current_ms() - start_time) + ","
                            + str(sklearn.metrics.confusion_matrix(y_test, y_pred).flatten()) + ","
                            + str(sklearn.metrics.accuracy_score(y_test, y_pred)) + ","
                            + str(sklearn.metrics.balanced_accuracy_score(y_test, y_pred)) + ","
                            + str(sklearn.metrics.matthews_corrcoef(y_test, y_pred)) + ","
                            + str(sklearn.metrics.log_loss(y_test, y_proba)) + "\n")
