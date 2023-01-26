import numpy
import pandas

import sklearn.model_selection as ms
from pyod.models.copod import COPOD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from mandalalib.EnsembleMetric import QStatMetric
from mandalalib.MEnsemble import MEnsemble

LABEL_NAME = 'multilabel'
CSV_FILE = "datasets/NSLKDD_Shuffled.csv"


def read_csv_dataset(dataset_name, label_name=LABEL_NAME, normal_tag="normal", limit=numpy.nan):
        """
        Method to process an input dataset as CSV
        :param normal_tag: tag that identifies normal data
        :param limit: integer to cut dataset if needed.
        :param dataset_name: name of the file (CSV) containing the dataset
        :param label_name: name of the feature containing the label
        :return: many values for analysis
        """
        # Loading Dataset
        df = pandas.read_csv(dataset_name, sep=",")

        # Shuffle
        df = df.sample(frac=1.0)
        df = df.fillna(0)
        df = df.replace('null', 0)

        # Testing Purposes
        if (numpy.isfinite(limit)) & (limit < len(df.index)):
            df = df[0:limit]

        # Binarize label
        y_enc = numpy.where(df[label_name] == normal_tag, 0, 1)

        # Basic Pre-Processing
        normal_frame = df.loc[df[label_name] == "normal"]
        print("\nDataset '" + dataset_name + "' loaded: " + str(len(df.index)) + " items, " + str(
            len(normal_frame.index)) + " normal and 2 labels")
        att_perc = (y_enc == 1).sum() / len(y_enc)

        # Train/Test Split of Classifiers
        x = df.drop(columns=[label_name])
        x_no_cat = x.select_dtypes(exclude=['object'])
        feature_list = x_no_cat.columns
        x_train, x_test, y_train, y_test = ms.train_test_split(x_no_cat, y_enc, test_size=0.5, shuffle=True)

        return x_train, x_test, y_train, y_test, feature_list, att_perc


def get_ensembles():
    return [
        MEnsemble(models_folder="",
                  classifiers=[DecisionTreeClassifier(), GaussianNB(), LinearDiscriminantAnalysis()],
                  diversity_metrics=[QStatMetric()],
                  bin_adj=LinearDiscriminantAnalysis()),
        MEnsemble(models_folder="",
                  classifiers=[DecisionTreeClassifier(), GaussianNB(), LinearDiscriminantAnalysis()],
                  diversity_metrics=[QStatMetric()],
                  bin_adj=DecisionTreeClassifier()),
        MEnsemble(models_folder="",
                  classifiers=[DecisionTreeClassifier(), LinearDiscriminantAnalysis()],
                  diversity_metrics=[QStatMetric()],
                  bin_adj=LinearDiscriminantAnalysis()),
        MEnsemble(models_folder="",
                  classifiers=[DecisionTreeClassifier(), LinearDiscriminantAnalysis()],
                  diversity_metrics=[QStatMetric()],
                  bin_adj=DecisionTreeClassifier()),
        MEnsemble(models_folder="",
                  classifiers=[DecisionTreeClassifier(), AdaBoostClassifier(), LinearDiscriminantAnalysis()],
                  diversity_metrics=[QStatMetric()],
                  bin_adj=LinearDiscriminantAnalysis()),
        MEnsemble(models_folder="",
                  classifiers=[DecisionTreeClassifier(), AdaBoostClassifier(), LinearDiscriminantAnalysis()],
                  diversity_metrics=[QStatMetric()],
                  bin_adj=DecisionTreeClassifier()),
        MEnsemble(models_folder="",
                  classifiers=[DecisionTreeClassifier(), COPOD(), LinearDiscriminantAnalysis()],
                  diversity_metrics=[QStatMetric()],
                  bin_adj=LinearDiscriminantAnalysis()),
        MEnsemble(models_folder="",
                  classifiers=[DecisionTreeClassifier(), COPOD(), LinearDiscriminantAnalysis()],
                  diversity_metrics=[QStatMetric()],
                  bin_adj=DecisionTreeClassifier()),
    ]


if __name__ == '__main__':

    # Reads CSV Dataset
    x_train, x_test, y_train, y_test, feature_list, att_perc = read_csv_dataset(CSV_FILE)

    # Loops over MEnsembles
    for m_ens in get_ensembles():

        # Exercises the Ensemble
        m_ens.fit(x_train, y_train, verbose=False)
        ens_predictions, adj_data, clf_predictions = m_ens.predict(x_test)

        # Reports MEnsemble Stats
        metric_scores, clf_metrics = m_ens.report(ens_predictions, clf_predictions, y_test, verbose=False)

        print("Ensemble Accuracy: " + str(clf_metrics["adj"]["acc"]) + " - QStat: " + str(metric_scores["QStat"]))




