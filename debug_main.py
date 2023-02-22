import numpy
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
from mandalalib.utils.MUtils import read_csv_dataset, read_csv_binary_dataset, get_clf_name, current_ms

LABEL_NAME = 'multilabel'
CSV_FILE = "datasets/AndMal_Shuffled.csv"
CSV_FOLDER = "datasets"
OUTPUT_FILE = "./output/bindatasets_V1.csv"

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


if __name__ == '__main__':

    with open(OUTPUT_FILE, 'w') as f:
        f.write('dataset,mandala_tag,clfs,')
        for dm in DIVERSITY_METRICS:
            f.write(dm.get_name() + ",")
        f.write('matrix,acc,b_acc,mcc,logloss,best_base_acc,best_base_mcc,acc_gain,mcc_gain,\n')

    for file in os.listdir(CSV_FOLDER):
        if file.endswith(".csv"):

            # Reads CSV Dataset
            x_train, x_test, y_train, y_test, feature_list, att_perc = \
                read_csv_binary_dataset(os.path.join(CSV_FOLDER, file))

            # Runs Basic Classifiers
            # b_clfs = []
            # for clf in [GaussianNB(), LinearDiscriminantAnalysis(), DecisionTreeClassifier()]:
            #     clf.fit(x_train, y_train)
            #     b_clfs.append(clf)
            #     y_pred = clf.predict(x_test)
            #     print(get_clf_name(clf) + " Accuracy: " + str(sklearn.metrics.balanced_accuracy_score(y_test, y_pred)))

            u_clfs = []
            cont = att_perc*2 if att_perc*2 < 0.5 else 0.5
            for clf in [UnsupervisedClassifier(COPOD(contamination=cont)),
                        UnsupervisedClassifier(IForest(contamination=cont, max_features=0.8, max_samples=0.8)),
                        UnsupervisedClassifier(HBOS(contamination=cont, n_bins=100)),
                        UnsupervisedClassifier(CBLOF(contamination=cont, alpha=0.75, beta=3))]:
                start_time = current_ms()
                clf.fit(x_train, y_train)
                u_clfs.append(clf)
                y_pred = clf.predict(x_test)
                print(get_clf_name(clf) + " Accuracy: " + str(
                    sklearn.metrics.accuracy_score(y_test, y_pred))
                      + " Train time: " + str(current_ms() - start_time) + " ms")

            # Runs Tree-Based Classifiers
            tb_clfs = []
            for clf in [XGB(), RandomForestClassifier()]:
                start_time = current_ms()
                clf.fit(x_train, y_train)
                tb_clfs.append(clf)
                y_pred = clf.predict(x_test)
                print(get_clf_name(clf) + " Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred))
                      + " Train time: " + str(current_ms() - start_time) + " ms")

            # Runs DL Tabular Classifiers
            dl_clfs = []
            for clf in [FastAI(feature_names=feature_list), TabNet(epochs=40, verbose=1, patience=2)]:
                start_time = current_ms()
                clf.fit(x_train, y_train)
                dl_clfs.append(clf)
                y_pred = clf.predict(x_test)
                print(get_clf_name(clf) + " Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred))
                      + " Train time: " + str(current_ms() - start_time) + " ms")

            # Loops over MEnsembles
            for m_ens in get_ensembles(u_clfs, tb_clfs, dl_clfs, cont):

                # Exercises the Ensemble
                start_time = current_ms()
                m_ens.fit(x_train, y_train, verbose=False)
                ens_predictions, adj_data, clf_predictions = m_ens.predict(x_test)

                # Reports MEnsemble Stats
                metric_scores, clf_metrics = m_ens.report(ens_predictions, clf_predictions, y_test, verbose=False)

                print(m_ens.get_name() + " Accuracy: " + str(clf_metrics["adj"]["acc"])
                      + " Time: " + str(current_ms() - start_time) + " ms")

                # Logging to file
                with open(OUTPUT_FILE, 'a') as f:
                    f.write(file + "," + m_ens.get_name() + "," + m_ens.get_clf_string() + ",")
                    for met in metric_scores.keys():
                        if isinstance(metric_scores[met], numpy.ndarray):
                            f.write(numpy.array2string(metric_scores[met], separator=' ').replace('\n', '') + ",")
                        else:
                            f.write(str(metric_scores[met]) + ",")
                    for met in clf_metrics["adj"].keys():
                        if isinstance(clf_metrics["adj"][met], numpy.ndarray):
                            f.write(numpy.array2string(clf_metrics["adj"][met], separator=' ').replace('\n', '') + ",")
                        else:
                            f.write(str(clf_metrics["adj"][met]) + ",")
                    f.write(",".join([str(x) for x in m_ens.feature_importance()]))
                    f.write('\n')
