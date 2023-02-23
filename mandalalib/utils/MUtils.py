import time

import numpy
import pandas
import sklearn

from mandalalib.classifiers.MANDALAClassifier import MANDALAClassifier


def read_csv_dataset(dataset_name, label_name="multilabel", limit=numpy.nan, split=True):
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


def read_csv_binary_dataset(dataset_name, label_name="multilabel", normal_tag="normal", limit=numpy.nan, split=True):
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
