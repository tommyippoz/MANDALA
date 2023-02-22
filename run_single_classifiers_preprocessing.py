import copy
import math

import numpy
import sklearn.metrics

import os

from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from mandalalib.EnsembleMetric import QStatMetric, SigmaMetric, CoupleDisagreementMetric, DisagreementMetric, \
    SharedFaultMetric
from mandalalib.classifiers.MANDALAClassifier import TabNet, FastAI, UnsupervisedClassifier, XGB
from mandalalib.utils.MUtils import read_csv_binary_dataset, get_clf_name, current_ms, read_csv_dataset

LABEL_NAME = 'multilabel'
CSV_FOLDER = "datasets_red"
OUTPUT_FILE = "./output/multiconf_single_multilabel_scores.csv"

DIVERSITY_METRICS = [QStatMetric(), SigmaMetric(), CoupleDisagreementMetric(), DisagreementMetric(),
                     SharedFaultMetric()]

if __name__ == '__main__':

    with open(OUTPUT_FILE, 'w') as f:
        f.write('dataset,clf,time,tag,matrix,acc,b_acc,mcc,logloss\n')

    for file in os.listdir(CSV_FOLDER):
        if file.endswith(".csv"):

            # Reads CSV Dataset
            x_train, x_test, y_train, y_test, feature_list, att_perc = \
                read_csv_dataset(os.path.join(CSV_FOLDER, file))

            # PCA
            pca_2 = PCA(n_components=math.ceil(x_train.shape[1] / 2.0)).fit(x_train)
            x_train_pca2 = pca_2.transform(x_train)
            x_test_pca2 = pca_2.transform(x_test)
            pca_4 = PCA(n_components=math.ceil(x_train.shape[1] / 4.0)).fit(x_train)
            x_train_pca4 = pca_4.transform(x_train)
            x_test_pca4 = pca_4.transform(x_test)

            # Scaling
            scaler = MinMaxScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.fit_transform(x_test)

            # PCA Scaling
            scaled_pca_2 = PCA(n_components=math.ceil(x_train.shape[1] / 2.0)).fit(x_train_scaled)
            x_train_scaled_pca2 = scaled_pca_2.transform(x_train_scaled)
            x_test_scaled_pca2 = scaled_pca_2.transform(x_test_scaled)
            scaled_pca_4 = PCA(n_components=math.ceil(x_train.shape[1] / 4.0)).fit(x_train_scaled)
            x_train_scaled_pca4 = scaled_pca_4.transform(x_train_scaled)
            x_test_scaled_pca4 = scaled_pca_4.transform(x_test_scaled)

            sets = {"Regular": [x_train, x_test],
                    "PCA_half": [x_train_pca2, x_test_pca2],
                    "PCA_quarter": [x_train_pca4, x_test_pca4],
                    "Scaled": [x_train_scaled, x_test_scaled],
                    "PCA_half_scaled": [x_train_scaled_pca2, x_test_scaled_pca2],
                    "PCA_quarter_scaled": [x_train_scaled_pca4, x_test_scaled_pca4]}

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
            for my_clf in [GaussianNB(), LinearDiscriminantAnalysis(),
                           XGB(), RandomForestClassifier(), DecisionTreeClassifier()]:
                for set_key in sets.keys():
                    clf = copy.deepcopy(my_clf)
                    train_s = sets[set_key][0]
                    test_s = sets[set_key][1]
                    start_time = current_ms()
                    clf.fit(train_s, y_train)
                    tb_clfs.append(clf)
                    y_proba = clf.predict_proba(test_s)
                    y_pred = clf.predict(test_s)
                    print(get_clf_name(clf) + "[" + set_key + "] Accuracy: " +
                          str(sklearn.metrics.accuracy_score(y_test, y_pred))
                          + " Train time: " + str(current_ms() - start_time) + " ms")

                    # Logging to file
                    with open(OUTPUT_FILE, 'a') as f:
                        f.write(file + "," + get_clf_name(clf) + "," + str(current_ms() - start_time) + "," + set_key + ","
                                + str(sklearn.metrics.confusion_matrix(y_test, y_pred).flatten()) + ","
                                + str(sklearn.metrics.accuracy_score(y_test, y_pred)) + ","
                                + str(sklearn.metrics.balanced_accuracy_score(y_test, y_pred)) + ","
                                + str(sklearn.metrics.matthews_corrcoef(y_test, y_pred)) + ","
                                + str(sklearn.metrics.log_loss(y_test, y_proba)) + "\n")

            # Runs DL Tabular Classifiers
            dl_clfs = []
            for set_key in sets.keys():
                train_s = sets[set_key][0]
                test_s = sets[set_key][1]
                for clf in [FastAI(),
                            TabNet(epochs=40, verbose=0, patience=2)]:
                    start_time = current_ms()
                    clf.fit(train_s, y_train)
                    dl_clfs.append(clf)
                    y_proba = clf.predict_proba(test_s)
                    y_pred = clf.predict(test_s)
                    print(get_clf_name(clf) + "[" + set_key + "] Accuracy: " +
                          str(sklearn.metrics.accuracy_score(y_test, y_pred))
                          + " Train time: " + str(current_ms() - start_time) + " ms")

                    # Logging to file
                    with open(OUTPUT_FILE, 'a') as f:
                        f.write(
                            file + "," + get_clf_name(clf) + "," + str(current_ms() - start_time) + "," + set_key + ","
                            + str(sklearn.metrics.confusion_matrix(y_test, y_pred).flatten()) + ","
                            + str(sklearn.metrics.accuracy_score(y_test, y_pred)) + ","
                            + str(sklearn.metrics.balanced_accuracy_score(y_test, y_pred)) + ","
                            + str(sklearn.metrics.matthews_corrcoef(y_test, y_pred)) + ","
                            + str(sklearn.metrics.log_loss(y_test, y_proba)) + "\n")

                    clf = None
