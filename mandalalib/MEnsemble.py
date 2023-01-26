import copy
import time

import numpy
import pandas
import sklearn


class MEnsemble:

    def __init__(self, models_folder, classifiers=[], diversity_metrics=[], bin_adj=[],
                 use_training=True, store_data=True):
        """
        Constructor for the MEnsemble object
        """
        self.classifiers = classifiers
        self.diversity_metrics = diversity_metrics
        self.store_data = store_data
        self.use_training = use_training
        self.models_folder = models_folder
        self.adj_data = {"train": {}, "test": {}}
        self.binary_adjudicator = bin_adj

    def add_classifier(self, clf):
        """
        Method to add a classifier
        :param clf: the classifier to add to the ensemble
        :return:
        """
        self.classifiers.append(clf)

    def set_adjudicator(self, clf):
        """
        Method to set the binary adjudicator
        :param clf: the classifier to add as a binary adjudicator
        :return:
        """
        self.binary_adjudicator = clf

    def fit(self, train_x, train_y, verbose=True):
        """
        Method to train the MEnsemble
        :param train_x: train set
        :param train_y: train labels
        :param verbose: true if debug information needs to be shown
        :return:
        """
        if verbose:
            print("Classifiers will be trained using %d train data points" % (len(train_y)))

        adj_data = []
        adj_features = []
        for clf in self.classifiers:
            clf_name = clf.__class__.__name__
            try:
                start = time.time()
                clf.fit(train_x, train_y)
                clf_pred = clf.predict_proba(train_x)
            except:
                print("Execution of learner " + clf_name + " failed")
                clf_pred = numpy.full((len(train_y), 2), 0.5)

            adj_data.append(clf_pred)
            bin_pred = [[0 if clf_pred[i][0] >= clf_pred[i][1] else 1]
                             for i in range(len(clf_pred))]
            adj_data.append(bin_pred)
            adj_features.extend([clf_name + "_normal",
                                 clf_name + "_anomaly",
                                 clf_name + "_label"])
            if verbose:
                print("Training of classifier '" + clf_name + "' completed in " +
                      str(time.time() - start) + " seconds, train accuracy " +
                      str(sklearn.metrics.accuracy_score(train_y, bin_pred)))

        # Cleans up and unifies the dataset of predictions
        adj_data = numpy.concatenate(adj_data, axis=1)
        adj_data = numpy.nan_to_num(adj_data, nan=-0.5, posinf=0.5, neginf=0.5)

        if self.use_training:
            adj_data = numpy.concatenate([train_x, adj_data], axis=1)

        # Trains meta-level learner
        if self.binary_adjudicator is not None:
            start = time.time()
            self.binary_adjudicator.fit(adj_data, train_y)
            adj_pred = self.binary_adjudicator.predict(adj_data)
            if verbose:
                print("Training of adjudicator '" + self.binary_adjudicator.__class__.__name__ +
                      "' completed in " + str(time.time() - start) + " seconds, train accuracy " +
                      str(sklearn.metrics.accuracy_score(train_y, adj_pred)))

        # Stores pandas DF if needed
        if self.store_data:
            self.adj_data["train"]["x"] = copy.deepcopy(adj_data)
            self.adj_data["train"]["y"] = train_y

    def predict(self, test_x):
        """
        Predict function to match SKLEARN standards
        :param test_x: test set
        :return: predictions of the ensemble,
                    data used to compute them and
                        predictions of individual classifiers
        """
        adj_data = []
        clf_predictions = []

        for clf in self.classifiers:
            clf_name = clf.__class__.__name__
            try:
                start = time.time()
                clf_pred = clf.predict_proba(test_x)
            except:
                print("Execution of learner " + clf_name + " failed")
                clf_pred = numpy.full((test_x.shape[0], 2), 0.5)

            adj_data.append(clf_pred)
            bin_pred = [[0 if clf_pred[i][0] >= clf_pred[i][1] else 1]
                        for i in range(len(clf_pred))]
            adj_data.append(bin_pred)
            clf_predictions.append(bin_pred)

        # Cleans up and unifies the dataset of predictions
        adj_data = numpy.concatenate(adj_data, axis=1)
        adj_data = numpy.nan_to_num(adj_data, nan=-0.5, posinf=0.5, neginf=0.5)
        clf_predictions = numpy.concatenate(clf_predictions, axis=1)
        clf_predictions = numpy.nan_to_num(clf_predictions, nan=-0, posinf=0, neginf=0)

        if self.use_training:
            adj_data = numpy.concatenate([test_x, adj_data], axis=1)

        # Stores pandas DF if needed
        if self.store_data:
            self.adj_data["test"]["x"] = copy.deepcopy(adj_data)

        return self.binary_adjudicator.predict(adj_data), adj_data, clf_predictions


    def report(self, adj_scores, clf_predictions, test_y, verbose=True):
        """

        :param adj_scores: scores of the clfs in the ensemble
        :param clf_predictions: predictions of the clfs in the ensemble
        :param test_y: labels of the test set
        :param verbose: True if debug information need to be shown
        :return:
        """
        metric_scores = {}
        for metric in self.diversity_metrics:
            metric_scores[metric.get_name()] = metric.compute_diversity(clf_predictions, test_y)
            if verbose:
                print("Diversity using metric " + metric.get_name() + ": " + str(metric_scores[metric.get_name()]))

        clf_metrics = {}
        for i in range(0, clf_predictions.shape[1]):
            clf_metrics["clf_" + str(i)] = {}
            clf_metrics["clf_" + str(i)]["matrix"] = sklearn.metrics.confusion_matrix(test_y, clf_predictions[:, i])
            clf_metrics["clf_" + str(i)]["f1"] = sklearn.metrics.f1_score(test_y, clf_predictions[:, i])
            clf_metrics["clf_" + str(i)]["acc"] = sklearn.metrics.accuracy_score(test_y, clf_predictions[:, i])
            clf_metrics["clf_" + str(i)]["mcc"] = sklearn.metrics.matthews_corrcoef(test_y, clf_predictions[:, i])
        clf_metrics["adj"] = {}
        clf_metrics["adj"]["matrix"] = sklearn.metrics.confusion_matrix(test_y, adj_scores)
        clf_metrics["adj"]["f1"] = sklearn.metrics.f1_score(test_y, adj_scores)
        clf_metrics["adj"]["acc"] = sklearn.metrics.accuracy_score(test_y, adj_scores)
        clf_metrics["adj"]["mcc"] = sklearn.metrics.matthews_corrcoef(test_y, adj_scores)

        return metric_scores, clf_metrics

    def get_name(self):
        pass



