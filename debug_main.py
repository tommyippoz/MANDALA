import sklearn.metrics

from pyod.models.copod import COPOD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRFClassifier

from mandalalib.EnsembleMetric import QStatMetric, SigmaMetric, CoupleDisagreementMetric, DisagreementMetric, \
    SharedFaultMetric
from mandalalib.MEnsemble import MEnsemble
from mandalalib.classifiers.MANDALAClassifier import TabNet, FastAI, LogisticReg
from mandalalib.utils.MUtils import read_csv_dataset, read_csv_binary_dataset

LABEL_NAME = 'multilabel'
CSV_FILE = "datasets/AndMal_Shuffled.csv"
OUTPUT_FILE = "./output/preliminary_outputNKok.csv"

DIVERSITY_METRICS = [QStatMetric(), SigmaMetric(), CoupleDisagreementMetric(), DisagreementMetric(),
                     SharedFaultMetric()]


def get_ensembles(set1, set2, set3):
    e_list = []
    for c2 in set2:
        for c3 in set3:
            for adj in [DecisionTreeClassifier(), LinearDiscriminantAnalysis(),
                        XGBClassifier(), RandomForestClassifier(n_estimators=10)]:
                e_list.append(MEnsemble(models_folder="",
                                        classifiers=[c2, c3],
                                        diversity_metrics=DIVERSITY_METRICS,
                                        bin_adj=adj))
    for c1 in set1:
        for c2 in set2:
            for c3 in set3:
                for adj in [DecisionTreeClassifier(), LinearDiscriminantAnalysis(),
                            XGBClassifier(), RandomForestClassifier(n_estimators=10)]:
                    e_list.append(MEnsemble(models_folder="",
                                            classifiers=[c1, c2, c3],
                                            diversity_metrics=DIVERSITY_METRICS,
                                            bin_adj=adj))

    return e_list


if __name__ == '__main__':

    with open(OUTPUT_FILE, 'w') as f:
        f.write('tag,clfs,')
        for dm in DIVERSITY_METRICS:
            f.write(dm.get_name() + ",")
        f.write('matrix,acc,mcc,best_base_acc,best_base_mcc,acc_gain,mcc_gain,\n')

    # Reads CSV Dataset
    x_train, x_test, y_train, y_test, feature_list, att_perc = read_csv_dataset(CSV_FILE)

    # Runs Basic Classifiers
    b_clfs = []
    for clf in [GaussianNB(), LinearDiscriminantAnalysis(), DecisionTreeClassifier()]:
        clf.fit(x_train, y_train)
        b_clfs.append(clf)
        y_pred = clf.predict(x_test)
        print(clf.__class__.__name__ + " Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred)))

    # Runs Tree-Based Classifiers
    tb_clfs = []
    for clf in [XGBClassifier(), RandomForestClassifier()]:
        clf.fit(x_train, y_train)
        tb_clfs.append(clf)
        y_pred = clf.predict(x_test)
        print(clf.__class__.__name__ + " Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred)))

    # Runs DL Tabular Classifiers
    dl_clfs = []
    for clf in [FastAI(feature_names=feature_list), TabNet()]:
        clf.fit(x_train, y_train)
        dl_clfs.append(clf)
        y_pred = clf.predict(x_test)
        print(clf.__class__.__name__ + " Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred)))

    # Loops over MEnsembles
    for m_ens in get_ensembles(b_clfs, tb_clfs, dl_clfs):

        # Exercises the Ensemble
        m_ens.fit(x_train, y_train, verbose=False)
        ens_predictions, adj_data, clf_predictions = m_ens.predict(x_test)

        # Reports MEnsemble Stats
        metric_scores, clf_metrics = m_ens.report(ens_predictions, clf_predictions, y_test, verbose=False)

        print(m_ens.get_name() + " Accuracy: " + str(clf_metrics["adj"]["acc"]))

        # Logging to file
        with open(OUTPUT_FILE, 'a') as f:
            f.write(m_ens.get_name() + "," + m_ens.get_clf_string() + ",")
            for met in metric_scores.keys():
                f.write(str(metric_scores[met]) + ",")
            for met in clf_metrics["adj"].keys():
                f.write(str(clf_metrics["adj"][met]) + ",")
            f.write('\n')
