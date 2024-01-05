import copy
import os

import numpy
import pandas
import sklearn.metrics
from logitboost import LogitBoost
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from mandalalib.classifiers.MANDALAClassifier import XGB
from mandalalib.utils.MUtils import entropy, compute_metrics, make_ensemble_prediction, make_dataset_dict, \
    compute_permutation_feature_importances

INPUT_FOLDER = "dataset_scores"
OUTPUT_FOLDER = "./output"
OUTPUT_FILENAME = "3comb_featImp_nodataset.csv"
USE_DATASET = False

STACKERS = [  # ('best', "best"), ('avg', "avg"), ('maxp', "maxp"), ('voting', "voting"),
    ('c:dt', DecisionTreeClassifier()),
    ('c:lda', LinearDiscriminantAnalysis()),
    ('c:nb', Pipeline([('ss', MinMaxScaler()), ('GNB', GaussianNB())])),
    #('c:gb', GradientBoostingClassifier(n_estimators=10))
    ]

TAB_CLFS = ['XGBoost', 'RandomForestClassifier', 'ExtraTreesClassifier']
TAB_NN = ['FastAI', 'TabNet(50-256-5)', 'PyTorch-Tabular(GATE)', 'PyTorch-Tabular()']
IMG_NN = ['PDI(pca-28-50-128-0.2)', 'PDI(tsne-28-50-128-0.2)', 'PDITL(mobilenet-tsne-128-30-256-0.2)']


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
                names = numpy.asarray([col for col in train_dataset_df.columns if 'datasetfeature_' in col])
                if len(names) > 10:
                    select_k_best_classifier = SelectKBest(f_classif, k=5)
                    train_red = select_k_best_classifier.fit_transform(train_dataset_df[names], train_labels)
                    names = names[select_k_best_classifier.get_support()]
                dataset_features = {'train': train_dataset_df[names].to_numpy(),
                                    'test': test_dataset_df[names].to_numpy()}

            # Compute 3-clf ensemble
            it_i = 0
            max_i = len(STACKERS) * len(TAB_CLFS) * len(TAB_NN) * len(IMG_NN)
            for tab_clf_name in TAB_CLFS:
                clf_t = test_dataset_dict[tab_clf_name]
                clft_metrics = compute_metrics(tab_clf_name, clf_t, test_labels)
                for tab_nn_name in TAB_NN:
                    clf_tn = test_dataset_dict[tab_nn_name]
                    clftn_metrics = compute_metrics(tab_nn_name, clf_tn, test_labels)
                    for img_nn_name in IMG_NN:
                        clf_in = test_dataset_dict[img_nn_name]
                        clfin_metrics = compute_metrics(img_nn_name, clf_in, test_labels)
                        best_base = clftn_metrics if clftn_metrics['b_acc'] > clfin_metrics['b_acc'] else clfin_metrics
                        best_base = clft_metrics if clft_metrics['b_acc'] > best_base['b_acc'] else best_base
                        all_metrics = []
                        for (name, adj) in STACKERS:
                            it_i += 1
                            ens_tag = 'ME@' + str(name) + '@' + '-'.join(
                                [str(x) for x in [tab_clf_name, tab_nn_name, img_nn_name]])
                            if existing_exps is not None and (((existing_exps['dataset_name'] == dataset_name) &
                                                               (existing_exps['ens_clf'] == ens_tag)).any()):
                                print('[%d/%d] Skipping classifier %s, already in the results' %
                                      (it_i, max_i, ens_tag))

                            else:
                                #try:
                                stacker_3, meta_model = \
                                    make_ensemble_prediction({tab_clf_name: train_dataset_dict[tab_clf_name],
                                                              tab_nn_name: train_dataset_dict[tab_nn_name],
                                                              img_nn_name: train_dataset_dict[img_nn_name]},
                                                             {tab_clf_name: test_dataset_dict[tab_clf_name],
                                                              tab_nn_name: test_dataset_dict[tab_nn_name],
                                                              img_nn_name: test_dataset_dict[img_nn_name]},
                                                             adj,
                                                             train_labels,
                                                             dataset_features)
                                stack2_metrics = compute_metrics(ens_tag, stacker_3, test_labels)
                                feat_imp = None
                                if meta_model is not None:
                                    column_names = [x for x in train_dataset_df.columns
                                                    if (tab_clf_name in x) or (tab_nn_name in x) or (img_nn_name in x)]
                                    if USE_DATASET:
                                        dataset_column_names = list(copy.deepcopy(names))
                                        dataset_column_names.extend(column_names)
                                        column_names = dataset_column_names
                                    feat_imp = compute_permutation_feature_importances(meta_model[0],
                                                                                       meta_model[1],
                                                                                       test_labels,
                                                                                       [tab_clf_name, tab_nn_name, img_nn_name],
                                                                                       column_names)
                                all_metrics.append([dataset_name, best_base, stack2_metrics, feat_imp])
                                print("[%d/%d] %s with '%s': BACC: %.3f vs %.3f" %
                                      (it_i, max_i, ens_tag, name,
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
                                            (',,,,,,,,,,' if exp_metrics[3] is None else ','.join(
                                                [key for key in feat_imp])) +
                                            '\n')
                            with open(OUTPUT_FILE, 'a') as f:
                                f.write(to_print + '\n')
