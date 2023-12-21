import contextlib
import copy
import logging
import os
import sys
import warnings

import numpy
import numpy as np
import pandas
import pandas as pd
import sklearn.metrics
from autogluon.tabular import TabularPredictor
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabular import TabularModel
from pytorch_tabular.config import OptimizerConfig, TrainerConfig, DataConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig, GatedAdditiveTreeEnsembleConfig, NodeConfig, \
    TabNetModelConfig
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


class MANDALAClassifier:
    """
    Basic Abstract Class for Classifiers.
    Abstract methods are only the classifier_name, with many degrees of freedom in implementing them.
    Wraps implementations from different frameworks (if needed), sklearn and many deep learning utilities
    """

    def __init__(self, model):
        """
        Constructor of a generic Classifier
        :param model: model to be used as Classifier
        """
        self.model = model
        self.trained = False
        self._estimator_type = "classifier"
        self.classes_ = None
        self.feature_importances_ = None
        self.X_ = None
        self.y_ = None

    def fit(self, x_train, y_train=None):
        """
        Fits a Classifier
        :param x_train: feature set
        :param y_train: labels
        """
        if y_train is not None:
            if isinstance(x_train, pd.DataFrame):
                self.model.fit(x_train.to_numpy(), y_train)
            else:
                self.model.fit(x_train, y_train)
            self.classes_ = numpy.unique(y_train)
        else:
            if isinstance(x_train, pd.DataFrame):
                self.model.fit(x_train.to_numpy())
            else:
                self.model.fit(x_train)
            self.classes_ = 2
        self.feature_importances_ = self.compute_feature_importances()
        self.trained = True

    def is_trained(self):
        """
        Flags if train was executed
        :return: True if trained, False otherwise
        """
        return self.trained

    def predict(self, x_test):
        """
        Method to compute predict of a classifier
        :return: array of predicted class
        """
        if isinstance(x_test, pandas.DataFrame):
            x_t = x_test.to_numpy()
        else:
            x_t = x_test
        return self.model.predict(x_t)

    def predict_proba(self, x_test):
        """
        Method to compute probabilities of predicted classes
        :return: array of probabilities for each classes
        """
        return self.model.predict_proba(x_test)

    def predict_confidence(self, x_test):
        """
        Method to compute confidence in the predicted class
        :return: -1 as default, value if algorithm is from framework PYOD
        """
        return -1

    def compute_feature_importances(self):
        """
        Outputs feature ranking in building a Classifier
        :return: ndarray containing feature ranks
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return numpy.sum(numpy.absolute(self.model.coef_), axis=0)
        return []

    def classifier_name(self):
        """
        Returns the name of the classifier (as string)
        """
        pass


class UnsupervisedClassifier(MANDALAClassifier):
    """
    Wrapper for unsupervised classifiers belonging to the library PYOD
    """

    def __init__(self, classifier):
        MANDALAClassifier.__init__(self, classifier)
        self.name = classifier.__class__.__name__
        self.revert = False

    def fit(self, x_train, y_train):
        super().fit(x_train)
        y_pred = self.predict(x_train)
        train_acc = sklearn.metrics.accuracy_score(y_train, y_pred)
        if train_acc < 0.5:
            self.revert = True

    def predict_proba(self, x_test):
        test_features = x_test.to_numpy() if isinstance(x_test, pandas.DataFrame) else x_test
        proba = self.model.predict_proba(test_features)
        pred = self.model.predict(test_features)
        for i in range(len(pred)):
            min_p = min(proba[i])
            max_p = max(proba[i])
            proba[i][pred[i]] = max_p
            proba[i][1 - pred[i]] = min_p
        if self.revert:
            proba[:] = proba[:, [1, 0]]
        return proba

    def predict(self, x_test):
        y_pred = super().predict(x_test)
        if self.revert:
            y_pred = abs(1-y_pred)
        return y_pred

    def predict_confidence(self, x_test):
        """
        Method to compute confidence in the predicted class
        :return: -1 as default, value if algorithm is from framework PYOD
        """
        return self.model.predict_confidence(x_test)

    def classifier_name(self):
        return self.name


class XGB(MANDALAClassifier):
    """
    Wrapper for the XGBoost  algorithm from xgboost library
    """

    def __init__(self, n_estimators=None, metric=None):
        self.metric = metric
        if n_estimators is None:
            MANDALAClassifier.__init__(self, XGBClassifier(use_label_encoder=False,
                                                           eval_metric=(
                                                               self.metric if self.metric is not None else "logloss")))
        else:
            MANDALAClassifier.__init__(self, XGBClassifier(n_estimators=n_estimators, use_label_encoder=False,
                                                           eval_metric=(
                                                               self.metric if self.metric is not None else "logloss")))

    def save_model(self, filename):
        self.model.save_model(filename)

    def classifier_name(self):
        return "XGBoost"


class TabNet(MANDALAClassifier):
    """
    Wrapper for the torch.tabnet algorithm
    """

    def __init__(self, metric=None, epochs=10, bsize=1024, patience=2, verbose=0):
        MANDALAClassifier.__init__(self, TabNetClassifier(verbose=verbose))
        self.epochs = epochs
        self.bsize = bsize
        self.patience = patience
        self.metric = metric if metric is not None else 'auc'

    def fit(self, x_train, y_train):
        if isinstance(x_train, pandas.DataFrame):
            x_t = x_train.to_numpy()
        else:
            x_t = copy.deepcopy(x_train)
        self.model.fit(X_train=x_t, y_train=y_train, max_epochs=self.epochs,
                       batch_size=self.bsize, eval_metric=[self.metric], patience=self.patience)
        self.classes_ = numpy.unique(y_train)
        self.feature_importances_ = self.compute_feature_importances()
        self.trained = True

    def predict(self, x_test):
        if isinstance(x_test, pandas.DataFrame):
            x_t = x_test.to_numpy()
        else:
            x_t = copy.deepcopy(x_test)
        return self.model.predict(x_t)

    def predict_proba(self, x_test):
        if isinstance(x_test, pandas.DataFrame):
            x_t = x_test.to_numpy()
        else:
            x_t = copy.deepcopy(x_test)
        return self.model.predict_proba(x_t)

    def classifier_name(self):
        return "TabNet(" + str(self.epochs) + "-" + str(self.bsize) + "-" + str(self.patience) + ")"


class AutoGluon(MANDALAClassifier):
    """
    Wrapper for classifiers taken from Gluon library
    clf_name options are
    ‘GBM’ (LightGBM)
    ‘CAT’ (CatBoost)
    ‘XGB’ (XGBoost)
    ‘RF’ (random forest)
    ‘XT’ (extremely randomized trees)
    ‘KNN’ (k-nearest neighbors)
    ‘LR’ (linear regression)
    ‘NN’ (neural network with MXNet backend)
    ‘FASTAI’ (neural network with FastAI backend)
    """

    def __init__(self, label_name, clf_name, metric, verbose=0):
        MANDALAClassifier.__init__(self, TabularPredictor(label=label_name, eval_metric=metric, verbosity=verbose))
        self.label_name = label_name
        self.clf_name = clf_name
        self.feature_importance = []

    def fit(self, x_train, y_train):
        self.feature_names = ["f" + str(i) for i in range(0, x_train.shape[1])]
        if isinstance(x_train, pd.DataFrame):
            df = pd.DataFrame(data=copy.deepcopy(x_train.values), columns=self.feature_names)
        else:
            df = pd.DataFrame(data=copy.deepcopy(x_train), columns=self.feature_names)
        df[self.label_name] = y_train
        self.model.fit(train_data=df, hyperparameters={self.clf_name: {}})
        self.feature_importance = self.model.feature_importance(df)
        self.feature_importances_ = self.compute_feature_importances()
        self.classes_ = numpy.unique(y_train)
        self.trained = True

    def compute_feature_importances(self):
        importances = []
        for feature in self.feature_names:
            if feature in self.feature_importance.importance.index.tolist():
                importances.append(abs(self.feature_importance.importance.get(feature)))
            else:
                importances.append(0.0)
        return np.asarray(importances)

    def predict(self, x_test):
        if isinstance(x_test, pd.DataFrame):
            df = pd.DataFrame(data=x_test.to_numpy(), columns=self.feature_names)
        else:
            df = pd.DataFrame(data=x_test, columns=self.feature_names)
        return self.model.predict(df, as_pandas=False)

    def predict_proba(self, x_test):
        if isinstance(x_test, pd.DataFrame):
            df = pd.DataFrame(data=x_test.to_numpy(), columns=self.feature_names)
        else:
            df = pd.DataFrame(data=x_test, columns=self.feature_names)
        return self.model.predict_proba(df, as_pandas=False)

    def classifier_name(self):
        return "AutoGluon"


class FastAI(AutoGluon):
    """
    Wrapper for the gluon.FastAI algorithm
    """

    def __init__(self, label_name="multilabel", metric="mcc", verbose=0):
        AutoGluon.__init__(self, label_name, "FASTAI", metric, verbose)

    def classifier_name(self):
        return "FastAI"


class GBM(AutoGluon):
    """
    Wrapper for the gluon.LightGBM algorithm
    """

    def __init__(self, feature_names, label_name, metric):
        AutoGluon.__init__(self, feature_names, label_name, "GBM", metric)

    def classifier_name(self):
        return "GBM"


class MXNet(AutoGluon):
    """
    Wrapper for the gluon.MXNet algorithm (to be debugged)
    """

    def __init__(self, feature_names, label_name):
        AutoGluon.__init__(self, feature_names, label_name, "NN")

    def classifier_name(self):
        return "MXNet"


class KNeighbors(MANDALAClassifier):
    """
    Wrapper for the sklearn.kNN algorithm
    """

    def __init__(self, k):
        MANDALAClassifier.__init__(self, KNeighborsClassifier(n_neighbors=k, n_jobs=-1, algorithm="kd_tree"))
        self.k = k

    def classifier_name(self):
        return str(self.k) + "NearestNeighbors"


class LogisticReg(MANDALAClassifier):
    """
    Wrapper for the sklearn.LogisticRegression algorithm
    """

    def __init__(self):
        MANDALAClassifier.__init__(self, LogisticRegression(solver='sag',
                                                            random_state=0,
                                                            multi_class='ovr',
                                                            max_iter=10000,
                                                            n_jobs=10,
                                                            tol=0.1))

    def classifier_name(self):
        return "LogisticRegression"


class PyTabularClassifier(MANDALAClassifier):
    """
        Wrapper for classifiers taken from Gluon library
        clf_name options are
        ‘GBM’ (LightGBM)
        ‘CAT’ (CatBoost)
        ‘XGB’ (XGBoost)
        ‘RF’ (random forest)
        ‘XT’ (extremely randomized trees)
        ‘KNN’ (k-nearest neighbors)
        ‘LR’ (linear regression)
        ‘NN’ (neural network with MXNet backend)
        ‘FASTAI’ (neural network with FastAI backend)
        """

    def __init__(self, label_name, clf_name, features, verbose=0):
        MANDALAClassifier.__init__(self, self.build_classifier(label_name, clf_name, features))
        self.label_name = label_name
        self.clf_name = clf_name
        self.feature_names = features
        self.feature_importance = []

    def build_classifier(self, label_name, clf_name, features):
        data_config = DataConfig(
            target=[label_name],
            # target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
            continuous_cols=list(features),
            categorical_cols=[],
            # num_workers=8
        )
        trainer_config = TrainerConfig(
            auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
            batch_size=1024,
            max_epochs=100,
        )
        optimizer_config = OptimizerConfig()

        if clf_name == 'NODE':
            model_config = NodeConfig(
                task="classification",
                depth=4,
                choice_function="entmax15",  # Number of nodes in each layer
                bin_function="entmoid15",  # Activation between each layers
                additional_tree_output_dim=3,
                input_dropout=0.0
            )
        elif clf_name == 'GATE':
            model_config = GatedAdditiveTreeEnsembleConfig(
                task="classification",
                num_trees=20,  # Number of nodes in each layer
                tree_depth=5,  # Activation between each layers
                chain_trees=False,
                share_head_weights=True,
            )
        elif clf_name == 'TabNet':
            model_config = TabNetModelConfig(
                task="classification",
                n_d=8,
                n_a=8,  # Number of nodes in each layer
                n_steps=3,  # Activation between each layers
                virtual_batch_size=128,
            )
        else:
            model_config = CategoryEmbeddingModelConfig(
                task="classification",
                layers="1024-512-512",  # Number of nodes in each layer
                activation="LeakyReLU",  # Activation between each layers
                learning_rate=1e-3,
            )

        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )

        return tabular_model

    def fit(self, x_train, y_train):
        if isinstance(x_train, pd.DataFrame):
            df = pd.DataFrame(data=x_train.to_numpy(), columns=self.feature_names)
        else:
            df = pd.DataFrame(data=x_train, columns=self.feature_names)
        df[self.label_name] = y_train
        self.model.fit(train=df)
        self.feature_importances_ = self.compute_feature_importances()
        self.classes_ = numpy.unique(y_train)
        self.trained = True

    def compute_feature_importances(self):
        importances = []

        return np.asarray(importances)

    def predict(self, x_test):
        if isinstance(x_test, pd.DataFrame):
            df = pd.DataFrame(data=x_test.to_numpy(), columns=self.feature_names)
        else:
            df = pd.DataFrame(data=x_test, columns=self.feature_names)
        pred_df = self.model.predict(df)
        return numpy.asarray(pred_df['prediction'])

    def predict_proba(self, x_test):
        if isinstance(x_test, pd.DataFrame):
            df = pd.DataFrame(data=x_test.to_numpy(), columns=self.feature_names)
        else:
            df = pd.DataFrame(data=x_test, columns=self.feature_names)
        pred_df = self.model.predict(df)
        df2 = pred_df.filter(regex='_probability')
        return df2.to_numpy()

    def classifier_name(self):
        return "PyTorch-Tabular(" + str(self.clf_name) + ")"

