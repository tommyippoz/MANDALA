import os

import sklearn.metrics
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
from mandalalib.classifiers.MANDALAClassifier import FastAI, XGB, PyTabularClassifier
from mandalalib.utils.MUtils import get_clf_name, current_ms, read_csv_dataset

LABEL_NAME = 'multilabel'
CSV_FOLDER = "datasets"
OUTPUT_FILE = "./output/single_multilabel_scores.csv"

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

if __name__ == '__main__':

    with open(OUTPUT_FILE, 'w') as f:
        f.write('dataset,clf,train_time,test_time,model_size,train_size,test_size,n_feat,acc,b_acc,mcc\n')

    for file in os.listdir(CSV_FOLDER):
        if file.endswith(".csv"):

            # Reads CSV Dataset
            x_train, x_test, y_train, y_test, feature_list = \
                read_csv_dataset(os.path.join(CSV_FOLDER, file), limit=50000, encode=True)

            print('--------------------------------------'
                  '\n-------- DATASET ' + str(file) + ' ---------\n')

            NN_CLFS = [PyTabularClassifier(label_name=LABEL_NAME, clf_name='NODE', features=feature_list),
                       PyTabularClassifier(label_name=LABEL_NAME, clf_name='TabNet', features=feature_list),
                       PyTabularClassifier(label_name=LABEL_NAME, clf_name='GATE', features=feature_list),
                       PyTabularClassifier(label_name=LABEL_NAME, clf_name='', features=feature_list)]

            all_clfs = TAB_CLFS + NN_CLFS
            for clf in [FastAI(label_name=LABEL_NAME, metric='accuracy')]:
                # Train
                start_time = current_ms()
                clf.fit(x_train.to_numpy(), y_train)
                train_time = current_ms()

                # Test
                y_proba = clf.predict_proba(x_test.to_numpy())
                test_time = current_ms()
                y_pred = clf.predict(x_test.to_numpy())

                # Quantifying size of the model
                dump(clf, "clf_dump.bin", compress=9)
                size = os.stat("clf_dump.bin").st_size
                os.remove("clf_dump.bin")

                # Console
                print(get_clf_name(clf) + " Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred))
                      + " Train time: " + str(current_ms() - start_time) + " ms")

                # Logging to file
                with open(OUTPUT_FILE, 'a') as f:
                    f.write(file + "," + get_clf_name(clf) + "," + str(train_time - start_time) + ","
                            + str(test_time - train_time) + "," + str(size) + ","
                            + str(len(y_train)) + "," + str(len(y_test)) + "," + str(len(feature_list)) + ","
                            + str(sklearn.metrics.accuracy_score(y_test, y_pred)) + ","
                            + str(sklearn.metrics.balanced_accuracy_score(y_test, y_pred)) + ","
                            + str(sklearn.metrics.matthews_corrcoef(y_test, y_pred)) + "\n")
