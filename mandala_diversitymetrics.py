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
            train_dataset_df = pandas.read_csv(os.path.join(INPUT_FOLDER, 'train', file))
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
                    similarity_metrics[dm.get_name()] = numpy.zeros((len(clf_names), len(clf_names)), dtype='float32')
            clf_names = list(dataset_dict.keys())
            for i in tqdm(range(0, len(clf_names)), "clfs"):
                clf_1 = dataset_dict[clf_names[i]]['label']
                for j in range(i + 1, len(clf_names)):
                    clf_2 = dataset_dict[clf_names[j]]['label']
                    couples = numpy.vstack([clf_1, clf_2]).T
                    for dm in DIVERSITY_METRICS:
                        similarity_metrics[dm.get_name()][i][j] += dm.compute_diversity(couples, label)

    # Printing average distance metrics
    for dm in DIVERSITY_METRICS:
        similarity_metrics[dm.get_name()] /= available_datasets
        df = pandas.DataFrame(data=similarity_metrics[dm.get_name()], columns=clf_names)
        df["clf_name"] = clf_names
        df.set_index('clf_name')
        df.to_csv(os.path.join(OUTPUT_FOLDER, dm.get_name() + ".csv"), index=True)
