import copy
import numpy
import os

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from mandalalib.EnsembleMetric import QStatMetric, SigmaMetric, CoupleDisagreementMetric, DisagreementMetric, \
    SharedFaultMetric
from mandalalib.MEnsemble import MEnsemble
from mandalalib.classifiers.MANDALAClassifier import TabNet, FastAI, XGB
from mandalalib.classifiers.PDITLClassifier import PDITLClassifier
from mandalalib.utils.MUtils import read_csv_binary_dataset, current_ms, report

LABEL_NAME = 'multilabel'
CSV_FOLDER = "datasets_red"
OUTPUT_FILE = "./output/full3.csv"

DIVERSITY_METRICS = [QStatMetric(), SigmaMetric(), CoupleDisagreementMetric(), DisagreementMetric(),
                     SharedFaultMetric(relative=False), SharedFaultMetric(relative=True)]


def get_ensembles(set1, set2, set3):
    e_list = []
    stackers = [DecisionTreeClassifier(), LinearDiscriminantAnalysis(),
                RandomForestClassifier(n_estimators=10)]
    for c1 in set1:
        for c2 in set2:
            for c3 in set3:
                for adj in stackers:
                    e_list.append(MEnsemble(models_folder="",
                                            classifiers=[copy.deepcopy(c1), copy.deepcopy(c2), copy.deepcopy(c3)],
                                            diversity_metrics=DIVERSITY_METRICS,
                                            bin_adj=adj,
                                            use_training=True))

    return e_list


if __name__ == '__main__':

    with open(OUTPUT_FILE, 'w') as f:
        f.write('dataset,mandala_tag,clfs,')
        for dm in DIVERSITY_METRICS:
            f.write(dm.get_name() + ",")
        f.write('matrix,acc,b_acc,mcc,logloss,best_base_acc,best_base_mcc,acc_gain,mcc_gain,'
                'train_time,test_time,max_bl_train,max_bl_test,train_bl,test_bl,tr_overhead,te_overhead,\n')

    for file in os.listdir(CSV_FOLDER):
        if file.endswith(".csv"):

            # Reads CSV Dataset
            x_train, x_test, y_train, y_test, feature_list, att_perc = \
                read_csv_binary_dataset(os.path.join(CSV_FOLDER, file), limit=5000)

            # Families of algorithms
            tb_clfs = [RandomForestClassifier(), XGB(), DecisionTreeClassifier()]
            dl_clfs = [FastAI(), TabNet(epochs=40, metric="mcc", verbose=0, patience=10)]
            pdi_clfs = [PDITLClassifier(n_classes=len(numpy.unique(y_train)), img_size=40,
                                        pdi_strategy='pca', epochs=50, bsize=1024, val_split=0.3, verbose=0),
                        PDITLClassifier(n_classes=len(numpy.unique(y_train)), img_size=70,
                                        pdi_strategy='pca', epochs=50, bsize=1024, val_split=0.3, verbose=0),
                        PDITLClassifier(n_classes=len(numpy.unique(y_train)), img_size=70,
                                        pdi_strategy='tsne', epochs=50, bsize=1024, val_split=0.3, verbose=0)
                        ]

            # Loops over MEnsembles
            for m_ens in get_ensembles(tb_clfs, dl_clfs, pdi_clfs):

                print("Ensemble " + m_ens.get_name())

                try:

                    # Exercises the Ensemble
                    start_time = current_ms()
                    max_bl_train, bl_train = m_ens.fit(x_train, y_train, verbose=True)
                    elapsed_time = current_ms()
                    elapsed_tr = elapsed_time - start_time
                    ens_predictions, adj_data, clf_predictions, max_bl_test, bl_test = m_ens.predict(x_test)
                    elapsed_te = current_ms() - elapsed_time

                    # Reports MEnsemble Stats
                    metric_scores, clf_metrics = report(ens_predictions, clf_predictions, y_test,
                                                        diversity_metrics=DIVERSITY_METRICS, verbose=False)
                    clf_metrics["adj"]["train_time"] = elapsed_tr
                    clf_metrics["adj"]["test_time"] = elapsed_te
                    clf_metrics["adj"]["max_bl_train_time"] = max_bl_train
                    clf_metrics["adj"]["max_bl_test_time"] = max_bl_test
                    clf_metrics["adj"]["train_bl_time"] = bl_train
                    clf_metrics["adj"]["test_bl_time"] = bl_test
                    clf_metrics["adj"]["train_overhead"] = (elapsed_tr - bl_train)/elapsed_tr
                    clf_metrics["adj"]["test_overhead"] = (elapsed_te - bl_test) / elapsed_te

                    print(m_ens.get_name() + " MCC: " + str(clf_metrics["adj"]["mcc"])
                          + " Train Time: " + str(elapsed_tr) + " ms")

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
                        fi_arr = m_ens.feature_importance()
                        fi_ok = [sum(fi_arr[:-9])].expand(fi_arr[-9:])
                        f.write(",".join([str(x) for x in fi_ok]))
                        f.write('\n')

                except:
                    print("Algorithm Failed")
