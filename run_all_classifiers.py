import numpy
import pandas
import sklearn.metrics

import os

import torch
from joblib import dump
from logitboost import LogitBoost
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from mandalalib.EnsembleMetric import QStatMetric, SigmaMetric, CoupleDisagreementMetric, DisagreementMetric, \
    SharedFaultMetric
from mandalalib.classifiers.MANDALAClassifier import FastAI, XGB, PyTabularClassifier, \
    LogisticReg
from mandalalib.classifiers.PDIClassifier import PDIClassifier
from mandalalib.classifiers.PDITLClassifier import PDITLClassifier
from mandalalib.utils.MUtils import read_csv_binary_dataset, get_clf_name, current_ms, read_csv_dataset

LABEL_NAME = 'multilabel'
CSV_FOLDER = "datasets_new/a"
OUTPUT_FILENAME = "single_multilabel_scores.csv"
OUTPUT_FOLDER = './output'

DIVERSITY_METRICS = [QStatMetric(), SigmaMetric(), CoupleDisagreementMetric(), DisagreementMetric(),
                     SharedFaultMetric()]

TAB_CLFS = [XGB(n_estimators=30),
            DecisionTreeClassifier(),
            Pipeline([("norm", MinMaxScaler()), ("gnb", GaussianNB())]),
            RandomForestClassifier(n_estimators=30),
            LinearDiscriminantAnalysis(),
            LogisticRegression(max_iter=10000),
            ExtraTreesClassifier(n_estimators=30),
            LogitBoost(n_estimators=30)]


def entropy(probs):
    norm_array = numpy.full(probs.shape[1], 1 / probs.shape[1])
    normalization = (-norm_array * numpy.log2(norm_array)).sum()
    ent = []
    for i in range(0, probs.shape[0]):
        val = numpy.delete(probs[i], numpy.where(probs[i] == 0))
        p = val / val.sum()
        ent.append(1 - (normalization - (-p * numpy.log2(p)).sum()) / normalization)
    return numpy.asarray(ent)


if __name__ == '__main__':

    OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
    with open(OUTPUT_FILE, 'w') as f:
        f.write('dataset,clf,train_time,test_time,model_size,train_size,test_size,n_feat,acc,b_acc,mcc\n')

    for file in os.listdir(CSV_FOLDER):
        if file.endswith(".csv"):

            # Reads CSV Dataset
            x_train, x_test, y_train, y_test, feature_list = \
                read_csv_dataset(os.path.join(CSV_FOLDER, file), limit=100000, encode=True)

            print('--------------------------------------'
                  '\n-------- DATASET ' + str(file) + ' ---------\n')

            NN_CLFS = [PyTabularClassifier(label_name=LABEL_NAME, clf_name='NODE', features=feature_list),
                       PyTabularClassifier(label_name=LABEL_NAME, clf_name='TabNet', features=feature_list),
                       PyTabularClassifier(label_name=LABEL_NAME, clf_name='GATE', features=feature_list),
                       PyTabularClassifier(label_name=LABEL_NAME, clf_name='', features=feature_list)]

            DI_CLFS = [PDIClassifier(n_classes=len(numpy.unique(y_train)), img_size=28,
                                     pdi_strategy='pca', epochs=50, bsize=128, val_split=0.2, verbose=2),
                       PDIClassifier(n_classes=len(numpy.unique(y_train)), img_size=28,
                                     pdi_strategy='tsne', epochs=50, bsize=128, val_split=0.2, verbose=2),
                       PDITLClassifier(n_classes=len(numpy.unique(y_train)), tl_tag='mobilenet', img_size=32,
                                       pdi_strategy='tsne', epochs=50, bsize=256, val_split=0.2, verbose=2),
                       PDITLClassifier(n_classes=len(numpy.unique(y_train)), tl_tag='mnist', img_size=28,
                                       pdi_strategy='tsne', epochs=50, bsize=256, val_split=0.2, verbose=2)
                       ]

            all_clfs = TAB_CLFS + NN_CLFS + DI_CLFS

            # Preparing df to output
            train_df = x_train.copy()
            train_df = train_df.add_prefix('datasetfeature_')
            test_df = x_test.copy()
            test_df = test_df.add_prefix('datasetfeature_')
            train_df[LABEL_NAME] = y_train
            test_df[LABEL_NAME] = y_test

            # Looping over classifiers
            for clf in all_clfs:
                clf_name = get_clf_name(clf)

                # Train
                start_time = current_ms()
                clf.fit(x_train.to_numpy(), y_train)
                train_time = current_ms()

                # Test
                y_pred = clf.predict(x_test.to_numpy())
                test_time = current_ms()

                # Quantifying size of the model
                dump(clf, "clf_dump.bin", compress=9)
                size = os.stat("clf_dump.bin").st_size
                os.remove("clf_dump.bin")

                # Console
                print(clf_name + " Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred))
                      + " Train time: " + str(current_ms() - start_time) + " ms")

                # Logging to file
                with open(OUTPUT_FILE, 'a') as f:
                    f.write(file + "," + clf_name + "," + str(train_time - start_time) + ","
                            + str(test_time - train_time) + "," + str(size) + ","
                            + str(len(y_train)) + "," + str(len(y_test)) + "," + str(len(feature_list)) + ","
                            + str(sklearn.metrics.accuracy_score(y_test, y_pred)) + ","
                            + str(sklearn.metrics.balanced_accuracy_score(y_test, y_pred)) + ","
                            + str(sklearn.metrics.matthews_corrcoef(y_test, y_pred)) + "\n")

                # Updating dataframes
                train_proba = clf.predict_proba(x_train.to_numpy())
                train_df[clf_name + '_label'] = clf.predict(x_train.to_numpy())
                train_df[clf_name + '_ent'] = entropy(train_proba)
                train_df[clf_name + '_maxp'] = numpy.max(train_proba, axis=1)
                for i in range(0, train_proba.shape[1]):
                    train_df[clf_name + '_prob' + str(i)] = train_proba[:, i]
                test_proba = clf.predict_proba(x_test.to_numpy())
                test_df[clf_name + '_label'] = y_pred
                test_df[clf_name + '_ent'] = entropy(test_proba)
                test_df[clf_name + '_maxp'] = numpy.max(test_proba, axis=1)
                for i in range(0, test_proba.shape[1]):
                    test_df[clf_name + '_prob' + str(i)] = test_proba[:, i]

            # Preparing df to output
            train_df[LABEL_NAME] = y_train
            test_df[LABEL_NAME] = y_test
            train_df.to_csv(os.path.join(OUTPUT_FOLDER, 'TRAIN_' + file), index=False)
            test_df.to_csv(os.path.join(OUTPUT_FOLDER, 'TEST_' + file), index=False)
