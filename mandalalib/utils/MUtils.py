import time

import numpy
import pandas
import sklearn

from mandalalib.classifiers.MANDALAClassifier import MANDALAClassifier


def read_csv_dataset(dataset_name, label_name="multilabel", limit=numpy.nan, split=True, shuffle=True):
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
    if shuffle:
        df = df.sample(frac=1.0)
    df = df.fillna(0)
    df = df.replace('null', 0)
    df = df[df.columns[df.nunique() > 1]]

    # Testing Purposes
    if (numpy.isfinite(limit)) & (limit < len(df.index)):
        df = df[0:limit]

    if split:
        encoding = pandas.factorize(df[label_name])
        y_enc = encoding[0]
        labels = encoding[1]
    else:
        y_enc = df[label_name]

    # Basic Pre-Processing
    normal_frame = df.loc[df[label_name] == "normal"]
    print("\nDataset '" + dataset_name + "' loaded: " + str(len(df.index)) + " items, " + str(
        len(normal_frame.index)) + " normal and " + str(len(numpy.unique(df[label_name]))) + " labels")

    # Train/Test Split of Classifiers
    x = df.drop(columns=[label_name])
    x_no_cat = x.select_dtypes(exclude=['object'])
    feature_list = x_no_cat.columns

    if split:
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_no_cat, y_enc, test_size=0.5, shuffle=True)
        return x_train, x_test, y_train, y_test, feature_list, numpy.NaN
    else:
        return x_no_cat, y_enc, feature_list, numpy.NaN


def read_csv_binary_dataset(dataset_name, label_name="multilabel", normal_tag="normal",
                            limit=numpy.nan, split=True, remove_constant=True):
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
    if remove_constant:
        df = df[df.columns[df.nunique() > 1]]

    # Testing Purposes
    if (numpy.isfinite(limit)) & (limit < len(df.index)):
        df = df[0:limit]

    # Binarize label
    if split:
        y_enc = numpy.where(df[label_name] == normal_tag, 0, 1)
    else:
        y_enc = df[label_name]

    # Basic Pre-Processing
    normal_frame = df.loc[df[label_name] == "normal"]
    print("\nDataset '" + dataset_name + "' loaded: " + str(len(df.index)) + " items, " + str(
        len(normal_frame.index)) + " normal and 2 labels")
    att_perc = (y_enc == 1).sum() / len(y_enc)

    # Train/Test Split of Classifiers
    x = df.drop(columns=[label_name])
    x_no_cat = x.select_dtypes(exclude=['object'])
    feature_list = x_no_cat.columns

    if split:
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_no_cat, y_enc, test_size=0.5, shuffle=True)
        return x_train, x_test, y_train, y_test, feature_list, att_perc
    else:
        return x_no_cat, y_enc, feature_list, att_perc


def current_ms():
    """
    Reports the current time in milliseconds
    :return: long int
    """
    return round(time.time() * 1000)


def get_classifier_name(clf):
    """
    Gets the name of the classifier
    :param clf: classifier to get the name of
    :return: a string
    """
    if isinstance(clf, MANDALAClassifier):
        return clf.classifier_name()
    else:
        return clf.__class__.__name__


def write_dict(dict_obj, filename, header=None):
    """
    writes dict obj to file
    :param dict_obj: obj to write
    :param filename: file to create
    :param header: optional header of the file
    :return: None
    """
    with open(filename, 'w') as f:
        if header is not None:
            f.write("%s\n" % header)
        write_rec_dict(f, dict_obj, "")


def write_rec_dict(out_f, dict_obj, prequel):
    """
    writes dict obj to file
    :param dict_obj: obj to write
    :param out_f: file object to append data to
    :param prequel: optional prequel to put as header of each new row
    :return: None
    """
    if (type(dict_obj) is dict) or issubclass(type(dict_obj), dict):
        for key in dict_obj.keys():
            if (type(dict_obj[key]) is dict) or issubclass(type(dict_obj[key]), dict):
                if len(dict_obj[key]) > 10:
                    for inner in dict_obj[key].keys():
                        if (prequel is None) or (len(prequel) == 0):
                            out_f.write("%s,%s,%s\n" % (key, inner, dict_obj[key][inner]))
                        else:
                            out_f.write("%s,%s,%s,%s\n" % (prequel, key, inner, dict_obj[key][inner]))
                else:
                    prequel = prequel + "," + str(key) if (prequel is not None) and (len(prequel) > 0) else str(key)
                    write_rec_dict(out_f, dict_obj[key], prequel)
            elif type(dict_obj[key]) is list:
                item_count = 1
                for item in dict_obj[key]:
                    new_prequel = prequel + "," + str(key) + ",item" + str(item_count) \
                        if (prequel is not None) and (len(prequel) > 0) else str(key) + ",item" + str(item_count)
                    write_rec_dict(out_f, item, new_prequel)
                    item_count += 1
            else:
                if (prequel is None) or (len(prequel) == 0):
                    out_f.write("%s,%s\n" % (key, dict_obj[key]))
                else:
                    out_f.write("%s,%s,%s\n" % (prequel, key, dict_obj[key]))
    else:
        if (prequel is None) or (len(prequel) == 0):
            out_f.write("%s\n" % dict_obj)
        else:
            out_f.write("%s,%s\n" % (prequel, dict_obj))


def check_fitted(clf):
    if hasattr(clf, "classes_"):
        return True
    else:
        return False


def get_clf_name(clf):
    if hasattr(clf, "classifier_name"):
        return clf.classifier_name()
    else:
        return clf.__class__.__name__


def compute_feature_importances(clf):
    """
    Outputs feature ranking in building a Classifier
    :return: ndarray containing feature ranks
    """
    if hasattr(clf, 'feature_importances_'):
        return clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        return numpy.sum(numpy.absolute(clf.coef_), axis=0)
    elif isinstance(clf, MANDALAClassifier):
        return clf.compute_feature_importances()
    return []

def report(adj_scores, clf_predictions, test_y, diversity_metrics, verbose=True):
    """

    :param adj_scores: scores of the clfs in the ensemble
    :param clf_predictions: predictions of the clfs in the ensemble
    :param test_y: labels of the test set
    :param verbose: True if debug information need to be shown
    :return:
    """
    metric_scores = {}
    for metric in diversity_metrics:
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

def entropy(probs):
    norm_array = numpy.full(probs.shape[1], 1 / probs.shape[1])
    normalization = (-norm_array * numpy.log2(norm_array)).sum()
    ent = []
    for i in range(0, probs.shape[0]):
        val = numpy.delete(probs[i], numpy.where(probs[i] == 0))
        p = val / val.sum()
        ent.append(1 - (normalization - (-p * numpy.log2(p)).sum()) / normalization)
    return numpy.asarray(ent)