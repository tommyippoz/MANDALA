import copy
import numpy
import os

import pandas
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from mandalalib.EnsembleMetric import QStatMetric, SigmaMetric, CoupleDisagreementMetric, DisagreementMetric, \
    SharedFaultMetric
from mandalalib.MEnsemble import MEnsemble
from mandalalib.classifiers.MANDALAClassifier import TabNet, FastAI, XGB

from mandalalib.utils.MUtils import read_csv_binary_dataset, current_ms, report, compute_feature_importances, \
    get_classifier_name

LABEL_NAME = 'multilabel'
CSV_FOLDER = "datasets_merged"
OUTPUT_FILE = "./output/bindatasets_bigdata_deepcopy_stronglearners_xgb500.csv"

DIVERSITY_METRICS = [QStatMetric(), SigmaMetric(), CoupleDisagreementMetric(), DisagreementMetric(),
                     SharedFaultMetric(relative=False), SharedFaultMetric(relative=True)]

STACKERS = [XGB(n_trees=500)]

def create_big_train():

    available_datasets = []
    for file in os.listdir(CSV_FOLDER):
        if file.endswith(".csv"):
            available_datasets.append(file.replace("_train.csv", "").replace("_test.csv", ""))
    available_datasets = sorted(list(set(available_datasets)))

    train_data = None
    train_labels = None

    print("Building Train Data")

    for file in available_datasets:

        print("Dataset " + file)

        # Reads CSV Dataset
        x_train, y_train, feature_list, att_perc = \
            read_csv_binary_dataset(os.path.join(CSV_FOLDER, file + "_train.csv"), label_name='label', split=False)
        x_train = x_train.drop(columns=[c for c in x_train.columns if c.startswith('dataset_')])

        if train_data is None:
            train_data = x_train
            train_labels = y_train.to_numpy()
        else:
            train_data = pandas.concat([train_data, x_train])
            train_labels = numpy.hstack([train_labels, y_train.to_numpy()])

    train_data['label'] = train_labels
    train_data.to_csv('big_train.csv', index=False)


if __name__ == '__main__':

    available_datasets = []
    for file in os.listdir(CSV_FOLDER):
        if file.endswith(".csv"):
            available_datasets.append(file.replace("_train.csv", "").replace("_test.csv", ""))
    available_datasets = sorted(list(set(available_datasets)))

    train_data, train_labels, feature_list, att_perc = \
        read_csv_binary_dataset('big_train.csv', label_name='label', split=False)

    # Loops over MEnsembles "DecisionTree",
    for tree_alg in ["DecisionTree", "RandomForest", "XGB"]:

        tree_columns = [x for x in train_data.columns if x.startswith(tree_alg)]

        for dnn in ["FastAI", "TabNet(40"]:

            dnn_columns = [x for x in train_data.columns if x.startswith(dnn)]

            for pdi in ["PDI_CNN(mobilenet-tsne-40", "PDITL_CNN(mobilenet-tsne-40",
                        "PDITL_CNN(mobilenet-pca-40", "PDITL_CNN(mobilenet-pca-70"]:

                pdi_columns = [x for x in train_data.columns if x.startswith(pdi)]

                print("Using %s / %s / %s" % (tree_alg, dnn, pdi))

                filtered_columns = tree_columns + dnn_columns + pdi_columns
                clf_pred_columns = [x for x in filtered_columns if x.endswith("_pred")]

                d_tr = train_data[filtered_columns]

                for stacker in STACKERS:

                    adj = copy.deepcopy(stacker)
                    start_time = current_ms()
                    adj.fit(d_tr, train_labels)
                    train_time = current_ms() - start_time

                    print(get_classifier_name(adj) + ' trained in ' + str(train_time) + ' ms')

                    for dataset in available_datasets:

                        # Reads CSV Dataset
                        x_test, y_test, feature_list, att_perc = \
                            read_csv_binary_dataset(os.path.join(CSV_FOLDER, dataset + "_test.csv"),
                                                    label_name='label',
                                                    remove_constant=False,
                                                    split=False)

                        d_te = x_test[filtered_columns]
                        clf_predictions = x_test[clf_pred_columns].to_numpy()

                        start_time = current_ms()
                        ens_predictions = adj.predict(d_te)
                        end_time = current_ms()

                        # Reports MEnsemble Stats
                        metric_scores, clf_metrics = report(ens_predictions, clf_predictions,
                                                            y_test, DIVERSITY_METRICS, verbose=False)

                        clf_metrics["adj"]["train_time"] = train_time
                        clf_metrics["adj"]["test_time"] = end_time - start_time

                        print("Adjudicator " + str(get_classifier_name(adj))
                              + " MCC: " + str(clf_metrics["adj"]["mcc"])
                              + " Time: " + str(end_time - start_time) + " ms")

                        # Logging to file
                        with open(OUTPUT_FILE, 'a') as f:
                            f.write(dataset + "," + get_classifier_name(adj) + "," +
                                    str(tree_alg) + "-" + str(dnn) + "-" + str(pdi) + ",")
                            for met in metric_scores.keys():
                                if isinstance(metric_scores[met], numpy.ndarray):
                                    f.write(
                                        numpy.array2string(metric_scores[met], separator=' ').replace('\n', '') + ",")
                                else:
                                    f.write(str(metric_scores[met]) + ",")
                            for met in clf_metrics["adj"].keys():
                                if isinstance(clf_metrics["adj"][met], numpy.ndarray):
                                    f.write(
                                        numpy.array2string(clf_metrics["adj"][met], separator=' ').replace('\n',
                                                                                                           '') + ",")
                                else:
                                    f.write(str(clf_metrics["adj"][met]) + ",")
                            f.write(",".join([str(x) for x in compute_feature_importances(adj)]))
                            f.write('\n')
