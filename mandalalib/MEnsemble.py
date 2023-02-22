import copy
import time

import numpy
import pandas
import sklearn

from mandalalib.utils.MUtils import check_fitted, get_clf_name, compute_feature_importances


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
        self.train_classes = 0

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
        self.train_classes = len(numpy.unique(train_y))
        for clf in self.classifiers:
            clf_name = get_clf_name(clf)
            #try:
            start = time.time()
            if not check_fitted(clf):
                clf.fit(train_x, train_y)
                if verbose:
                    print("Training of classifier '" + clf_name + "' completed in " +
                          str(time.time() - start) + " seconds")
            clf_pred = clf.predict_proba(train_x)
            #except:
            #    print("Execution of learner " + clf_name + " failed")
            #    clf_pred = numpy.full((len(train_y), self.train_classes), 1.0/self.train_classes)

            adj_data.append(clf_pred)
            class_pred = numpy.argmax(clf_pred, axis=1)
            adj_data.append(class_pred)
            adj_features.extend([clf_name + "_p_c" + str(i) for i in range(0, self.train_classes)])
            adj_features.extend([clf_name + "_label"])


        # Cleans up and unifies the dataset of predictions
        adj_data = numpy.column_stack(adj_data)
        adj_data = numpy.nan_to_num(adj_data, nan=-1.0/self.train_classes, posinf=1.0/self.train_classes, neginf=1.0/self.train_classes)

        if self.use_training:
            adj_data = numpy.concatenate([train_x, adj_data], axis=1)

        # Trains meta-level learner
        if self.binary_adjudicator is not None:
            start = time.time()
            self.binary_adjudicator.fit(adj_data, train_y)
            adj_pred = self.binary_adjudicator.predict(adj_data)
            if verbose:
                print("Training of adjudicator '" + get_clf_name(self.binary_adjudicator) +
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
            clf_name = get_clf_name(clf)
            try:
                start = time.time()
                clf_pred = clf.predict_proba(test_x)
            except:
                print("Execution of learner " + clf_name + " failed")
                clf_pred = numpy.full((test_x.shape[0], self.train_classes), 1.0/self.train_classes)

            adj_data.append(clf_pred)
            class_pred = numpy.argmax(clf_pred, axis=1)
            adj_data.append(class_pred)
            clf_predictions.append(class_pred)

        # Cleans up and unifies the dataset of predictions
        adj_data = numpy.column_stack(adj_data)
        adj_data = numpy.nan_to_num(adj_data, nan=-1.0/self.train_classes, posinf=1.0/self.train_classes, neginf=1.0/self.train_classes)
        clf_predictions = numpy.column_stack(clf_predictions)
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
            clf_metrics["clf_" + str(i)]["acc"] = sklearn.metrics.accuracy_score(test_y, clf_predictions[:, i])
            clf_metrics["clf_" + str(i)]["b_acc"] = sklearn.metrics.balanced_accuracy_score(test_y, clf_predictions[:, i])
            clf_metrics["clf_" + str(i)]["mcc"] = sklearn.metrics.matthews_corrcoef(test_y, clf_predictions[:, i])
            clf_metrics["clf_" + str(i)]["logloss"] = sklearn.metrics.log_loss(test_y, clf_predictions[:, i])
        clf_metrics["adj"] = {}
        clf_metrics["adj"]["matrix"] = numpy.asarray(sklearn.metrics.confusion_matrix(test_y, adj_scores)).flatten()
        clf_metrics["adj"]["acc"] = sklearn.metrics.accuracy_score(test_y, adj_scores)
        clf_metrics["adj"]["b_acc"] = sklearn.metrics.balanced_accuracy_score(test_y, adj_scores)
        clf_metrics["adj"]["mcc"] = sklearn.metrics.matthews_corrcoef(test_y, adj_scores)
        clf_metrics["adj"]["logloss"] = sklearn.metrics.log_loss(test_y, adj_scores)
        clf_metrics["adj"]["best_base_acc"] = max([clf_metrics[k]["acc"] for k in clf_metrics.keys() if k not in ["adj"]])
        clf_metrics["adj"]["best_base_mcc"] = max([abs(clf_metrics[k]["mcc"]) for k in clf_metrics.keys() if k not in ["adj"]])
        clf_metrics["adj"]["acc_gain"] = clf_metrics["adj"]["acc"] - clf_metrics["adj"]["best_base_acc"]
        clf_metrics["adj"]["mcc_gain"] = abs(clf_metrics["adj"]["mcc"]) - clf_metrics["adj"]["best_base_mcc"]

        return metric_scores, clf_metrics

    def get_name(self):
        tag = ""
        for clf in self.classifiers:
            tag = tag + get_clf_name(clf)[0] + get_clf_name(clf)[-1]
        tag = get_clf_name(self.binary_adjudicator) + " [" + str(len(self.classifiers)) + " - " + tag + "]"
        if self.use_training:
            tag = tag + "@full"
        else:
            tag = tag + "@nodataset"
        return tag

    def get_clf_string(self):
        tag = "["
        for clf in self.classifiers:
            tag = tag + get_clf_name(clf) + ";"
        return tag[0:-1] + "]"

    def feature_importance(self):
        fi = numpy.asarray(compute_feature_importances(self.binary_adjudicator))
        return fi/sum(fi)

