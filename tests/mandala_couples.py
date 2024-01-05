import copy
import os

import numpy
import pandas
import sklearn.metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from mandalalib.classifiers.MANDALAClassifier import XGB
from mandalalib.utils.MUtils import entropy, compute_feature_importances, compute_permutation_feature_importances, \
    make_dataset_dict, compute_metrics, make_ensemble_prediction

INPUT_FOLDER = "dataset_scores"
OUTPUT_FOLDER = "./output"
OUTPUT_FILENAME = "2comb_wfeatImp.csv"

USE_DATASET = True

STACKERS = [  # ('best', "best"), ('avg', "avg"), ('maxp', "maxp"),
    ('c:dt', DecisionTreeClassifier()),
    ('c:lda', LinearDiscriminantAnalysis()),
    ('c:nb', Pipeline([('ss', MinMaxScaler()), ('GNB', GaussianNB())])),
    # ('c:rf', RandomForestClassifier(n_estimators=10)),
    # ('c:xgb', XGB(n_estimators=10))
]


if __name__ == '__main__':

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)

    existing_exps = None
    if os.path.exists(OUTPUT_FILE):
        existing_exps = pandas.read_csv(OUTPUT_FILE)
        existing_exps = existing_exps.loc[:, ['dataset_name', 'ens_clf']]

    clf_names = []
    for file in os.listdir(os.path.join(INPUT_FOLDER, 'train')):
        if file.endswith(".csv"):

            # Loading Train Dataset
            dataset_name = file.replace('TRAIN_', '').replace('.csv', '')
            train_dataset_df = pandas.read_csv(os.path.join(INPUT_FOLDER, 'train', file))
            train_labels = train_dataset_df['multilabel']
            print("\n[DATASET] " + dataset_name)

            # Loading clfs names (only once)
            if len(clf_names) == 0:
                for col_name in train_dataset_df.columns:
                    if (not col_name.startswith("datasetfeature_")) and (col_name.endswith("_label")):
                        clf_names.append(col_name.replace("_label", ""))

            # Make dicts
            train_dataset_dict = make_dataset_dict(clf_names, train_dataset_df)
            test_dataset_df = pandas.read_csv(os.path.join(INPUT_FOLDER, 'test', 'TEST_' + dataset_name + '.csv'))
            test_labels = test_dataset_df['multilabel']
            test_dataset_dict = make_dataset_dict(clf_names, test_dataset_df)

            # Set use_dataset
            dataset_features = None
            if USE_DATASET:
                names = [col for col in train_dataset_df.columns if 'datasetfeature_' in col]
                dataset_features = {'train': train_dataset_df[names].to_numpy(),
                                    'test': test_dataset_df[names].to_numpy()}

            # Compute 2-clf ensemble
            it_i = 0
            max_i = len(STACKERS) * len(clf_names) * (len(clf_names) + 1) / 2
            for i in range(0, len(clf_names)):
                clf_1 = test_dataset_dict[clf_names[i]]
                clf1_metrics = compute_metrics(clf_names[i], clf_1, test_labels)
                for j in range(i + 1, len(clf_names)):
                    clf_2 = test_dataset_dict[clf_names[j]]
                    clf2_metrics = compute_metrics(clf_names[j], clf_2, test_labels)
                    best_base = clf1_metrics if clf1_metrics['b_acc'] > clf2_metrics['b_acc'] else clf2_metrics
                    all_metrics = []
                    for (name, adj) in STACKERS:
                        it_i += 1
                        ens_tag = 'ME@' + str(name) + '@' + '-'.join([str(x) for x in [clf_names[i], clf_names[j]]])
                        if existing_exps is not None and (((existing_exps['dataset_name'] == dataset_name) &
                                                           (existing_exps['ens_clf'] == ens_tag)).any()):
                            print('%d/%d Skipping classifier %s, already in the results' % (it_i, max_i, ens_tag))

                        else:
                            #try:
                            stacker_2, meta_model = \
                                make_ensemble_prediction({clf_names[i]: train_dataset_dict[clf_names[i]],
                                                          clf_names[j]: train_dataset_dict[clf_names[j]]},
                                                         {clf_names[i]: test_dataset_dict[clf_names[i]],
                                                          clf_names[j]: test_dataset_dict[clf_names[j]]},
                                                         adj,
                                                         train_labels,
                                                         dataset_features)
                            stack2_metrics = compute_metrics(ens_tag, stacker_2, test_labels)
                            feat_imp = None
                            if meta_model is not None:
                                column_names = [x for x in train_dataset_df.columns
                                                if (clf_names[i] in x) or (clf_names[j] in x)]
                                if USE_DATASET:
                                    dataset_column_names = [x for x in train_dataset_df.columns
                                                    if 'datasetfeature' in x]
                                    dataset_column_names.extend(column_names)
                                    column_names = dataset_column_names
                                feat_imp = compute_permutation_feature_importances(meta_model[0],
                                                                                   meta_model[1],
                                                                                   test_labels,
                                                                                   [clf_names[i], clf_names[j]],
                                                                                   column_names)
                            all_metrics.append([dataset_name, best_base, stack2_metrics, feat_imp])
                            print("[%d/%d] %s-%s with '%s': BACC: %.3f vs %.3f" %
                                  (it_i, max_i, clf1_metrics['clf'], clf2_metrics['clf'], name,
                                   stack2_metrics['b_acc'], best_base['b_acc']))

                            #except:
                            #    print('oh no')

                    for exp_metrics in all_metrics:
                        to_print = exp_metrics[0] + ','
                        to_print += ','.join([str(x) for x in exp_metrics[1].values()]) + ','
                        to_print += ','.join([str(x) for x in exp_metrics[2].values()]) + ','
                        to_print += ','.join([str(exp_metrics[2][s] - exp_metrics[1][s])
                                              for s in ['acc', 'b_acc', 'mcc', 'avg_conf',
                                                        'avg_conf_hit', 'avg_conf_misc']]) + ','
                        to_print += ','.join([str((exp_metrics[2][s] - exp_metrics[1][s]) / exp_metrics[1][s] if
                                                  exp_metrics[1][s] > 0 else 0.0)
                                              for s in ['acc', 'b_acc', 'mcc', 'avg_conf',
                                                        'avg_conf_hit', 'avg_conf_misc']]) + ','
                        if exp_metrics[3] is not None:
                            to_print += ','.join([str(exp_metrics[3][key]) for key in exp_metrics[3]])
                        if not os.path.exists(OUTPUT_FILE):
                            with open(OUTPUT_FILE, 'w') as f:
                                f.write('dataset_name,' +
                                        ','.join(['bestbase_' + str(x) for x in exp_metrics[1].keys()]) + ',' +
                                        ','.join(['ens_' + str(x) for x in exp_metrics[2].keys()]) + ',' +
                                        ','.join(['gain_' + str(s) for s in
                                                  ['acc', 'b_acc', 'mcc', 'avg_conf',
                                                   'avg_conf_hit', 'avg_conf_miss']]) + ',' +
                                        ','.join(['gain_p_' + str(s) for s in
                                                  ['acc', 'b_acc', 'mcc', 'avg_conf',
                                                   'avg_conf_hit', 'avg_conf_miss']]) + ',' +
                                        (',,,,,,,,,,' if exp_metrics[3] is None else ','.join([key for key in feat_imp])) +
                                        '\n')
                        with open(OUTPUT_FILE, 'a') as f:
                            f.write(to_print + '\n')
