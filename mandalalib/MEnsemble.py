import copy
import time

import numpy
import pandas
import sklearn

from mandalalib.utils.MUtils import check_fitted, get_clf_name, compute_feature_importances, current_ms, entropy


def compute_baselearner_data(clf_name, preds):
    full_predictions = [numpy.argmax(preds, axis=1), numpy.max(preds, axis=1), entropy(preds)]
    features = [clf_name + "_pred", clf_name + "_maxprob", clf_name + "_entropy"]
    return features, numpy.asarray(full_predictions).T


class MEnsemble:

    def __init__(self, models_folder, classifiers=[], diversity_metrics=[], bin_adj=[],
                 use_training=True, store_data=False, train_base=True):
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
        self.train_base = train_base

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
        max_bl_time = 0
        start_time = current_ms()
        self.train_classes = len(numpy.unique(train_y))
        for clf in self.classifiers:
            clf_name = get_clf_name(clf)
            start_bl = current_ms()
            if self.train_base:
                clf.fit(train_x, train_y)
                tr_time = current_ms() - start_bl
                if tr_time > max_bl_time:
                    max_bl_time = tr_time
                if verbose:
                    print("Training of classifier '" + clf_name + "' completed in " +
                          str(tr_time) + " ms")
            clf_pred = clf.predict_proba(train_x)
            bl_names, bl_data = compute_baselearner_data(clf_name, clf_pred)
            adj_data.append(bl_data)

        bl_time = current_ms() - start_time

        # Cleans up and unifies the dataset of predictions
        adj_data = numpy.column_stack(adj_data)
        adj_data = numpy.nan_to_num(adj_data, nan=-1.0/self.train_classes, posinf=1.0/self.train_classes, neginf=1.0/self.train_classes)

        if self.use_training:
            adj_data = numpy.concatenate([train_x, adj_data], axis=1)

        # Trains meta-level learner
        if self.binary_adjudicator is not None:
            start = current_ms()
            self.binary_adjudicator.fit(adj_data, train_y)
            adj_pred = self.binary_adjudicator.predict(adj_data)
            if verbose:
                print("Training of adjudicator '" + get_clf_name(self.binary_adjudicator) +
                      "' completed in " + str(current_ms() - start) + " ms, train accuracy " +
                      str(sklearn.metrics.accuracy_score(train_y, adj_pred)))

        # Stores pandas DF if needed
        if self.store_data:
            self.adj_data["train"]["x"] = copy.deepcopy(adj_data)
            self.adj_data["train"]["y"] = train_y

        return max_bl_time, bl_time

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
        max_bl_time = 0
        start_time = current_ms()
        for clf in self.classifiers:
            clf_name = get_clf_name(clf)
            try:
                start = current_ms()
                clf_pred = clf.predict_proba(test_x)
                te_time = current_ms() - start
                if te_time > max_bl_time:
                    max_bl_time = te_time
            except:
                print("Execution of learner " + clf_name + " failed")
                clf_pred = numpy.full((test_x.shape[0], self.train_classes), 1.0/self.train_classes)

            bl_names, bl_data = compute_baselearner_data(clf_name, clf_pred)
            adj_data.append(bl_data)
            clf_predictions.append(bl_data[:, 0])
        bl_time = current_ms() - start_time

        # Cleans up and unifies the dataset of predictions
        adj_data = numpy.column_stack(adj_data)
        adj_data = numpy.nan_to_num(adj_data, nan=-1.0 / self.train_classes, posinf=1.0 / self.train_classes,
                                    neginf=1.0 / self.train_classes)

        # Cleans up and unifies the dataset of predictions
        clf_predictions = numpy.column_stack(clf_predictions)
        clf_predictions = numpy.nan_to_num(clf_predictions, nan=-0, posinf=0, neginf=0)

        if self.use_training:
            adj_data = numpy.concatenate([test_x, adj_data], axis=1)

        # Stores pandas DF if needed
        if self.store_data:
            self.adj_data["test"]["x"] = copy.deepcopy(adj_data)

        return self.binary_adjudicator.predict(adj_data), adj_data, clf_predictions, max_bl_time, bl_time

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

