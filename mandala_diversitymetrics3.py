import os

import numpy
import pandas
import sklearn.metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from mandalalib.EnsembleMetric import QStatMetric, SigmaMetric, CoupleDisagreementMetric, DisagreementMetric, \
    SharedFaultMetric
from mandalalib.classifiers.MANDALAClassifier import XGB

INPUT_FOLDER = "dataset_scores"
OUTPUT_FOLDER = "./output"

DIVERSITY_METRICS = [QStatMetric(), SigmaMetric(), CoupleDisagreementMetric(),
                     DisagreementMetric(relative=True), SharedFaultMetric(relative=True)]

TAB_CLFS = ['XGBoost', 'RandomForestClassifier', 'ExtraTreesClassifier']
TAB_NN = ['FastAI', 'TabNet(50-256-5)', 'PyTorch-Tabular(GATE)', 'PyTorch-Tabular()']
IMG_NN = ['PDI(pca-28-50-128-0.2)', 'PDI(tsne-28-50-128-0.2)', 'PDITL(mobilenet-tsne-128-30-256-0.2)']

if __name__ == '__main__':

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    clf_names = []
    similarity_metrics = {}
    available_datasets = 0
    for file in os.listdir(os.path.join(INPUT_FOLDER, 'train')):
        if file.endswith(".csv"):
            available_datasets += 1
            dataset_name = file.replace('TRAIN_', '').replace('.csv', '')
            train_dataset_df = pandas.read_csv(os.path.join(INPUT_FOLDER, 'train', file), nrows=10000)
            label = train_dataset_df['multilabel'].to_numpy()
            print("\n[DATASET] " + dataset_name)

            # Loading clfs names (only once)
            if len(clf_names) == 0:
                for col_name in train_dataset_df.columns:
                    if (not col_name.startswith("datasetfeature_")) and (col_name.endswith("_label")):
                        clf_names.append(col_name.replace("_label", ""))

            dataset_dict = {}
            for clf_name in clf_names:
                clf_dict = {'label': train_dataset_df[clf_name + '_label'],
                            'maxp': train_dataset_df[clf_name + '_maxp'],
                            'ent': train_dataset_df[clf_name + '_ent'],
                            'probas': numpy.asarray([train_dataset_df[x] for x in train_dataset_df.columns
                                                     if x.startswith(clf_name) and '_prob' in x]).T}
                dataset_dict[clf_name] = clf_dict

            # Compute Diversity metrics
            if len(similarity_metrics) == 0:
                for dm in DIVERSITY_METRICS:
                    similarity_metrics[dm.get_name()] = {}
            for tab_clf_name in tqdm(TAB_CLFS, 'tab'):
                for tab_nn_name in TAB_NN:
                    for img_nn_name in IMG_NN:
                        couples = numpy.vstack([dataset_dict[tab_clf_name]['label'],
                                                dataset_dict[tab_nn_name]['label'],
                                                dataset_dict[img_nn_name]['label']]).T
                        tag = tab_clf_name + "-" + tab_nn_name + "-" + img_nn_name
                        for dm in DIVERSITY_METRICS:
                            if tag not in similarity_metrics[dm.get_name()].keys():
                                similarity_metrics[dm.get_name()][tag] = 0
                            similarity_metrics[dm.get_name()][tag] += dm.compute_diversity(couples, label)

    # Printing average distance metrics
    df = None
    datalist = []
    for dm in DIVERSITY_METRICS:
        a = numpy.asarray(list(similarity_metrics[dm.get_name()].values()))
        datalist.append(a/available_datasets)
    sim_numpy = numpy.vstack(datalist).T
    df = pandas.DataFrame(data=sim_numpy, columns=list(similarity_metrics.keys()))
    df["tag"] = list(similarity_metrics[dm.get_name()].keys())
    df.to_csv(os.path.join(OUTPUT_FOLDER, "triples_metrics.csv"), index=True)
