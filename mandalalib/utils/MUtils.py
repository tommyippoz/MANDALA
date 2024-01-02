import copy
import time

import numpy
import pandas
import sklearn
from sklearn.inspection import permutation_importance

from mandalalib.classifiers.MANDALAClassifier import MANDALAClassifier


def read_csv_dataset(dataset_name, label_name="multilabel", limit=numpy.nan, encode=True, split=True, shuffle=True):
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

    if encode:
        encoding = pandas.factorize(df[label_name])
        y_enc = encoding[0]
        labels = encoding[1]
    else:
        y_enc = df[label_name].to_numpy()

    # Basic Pre-Processing
    normal_frame = df.loc[df[label_name] == "normal"]
    print("\nDataset '" + dataset_name + "' loaded: " + str(len(df.index)) + " items, " + str(
        len(normal_frame.index)) + " normal and " + str(len(numpy.unique(df[label_name]))) + " labels")

    # Train/Test Split of Classifiers
    x = df.drop(columns=[label_name])
    x_no_cat = x.select_dtypes(exclude=['object'])
    feature_list = x_no_cat.columns

    if split:
        x_train, x_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(x_no_cat, y_enc, test_size=0.5, shuffle=shuffle)
        return x_train, x_test, y_train, y_test, feature_list
    else:
        return x_no_cat, y_enc, feature_list


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


def get_clf_name(classifier):
    clf_name = classifier.classifier_name() if hasattr(classifier,
                                                       'classifier_name') else classifier.__class__.__name__
    if clf_name == 'Pipeline':
        keys = list(classifier.named_steps.keys())
        clf_name = str(keys) if len(keys) != 2 else str(keys[1]).upper()
    return clf_name


def compute_feature_importances(clf, x_test=None, y_test=None):
    """
    Outputs feature ranking in building a Classifier
    :return: ndarray containing feature ranks
    """
    fi = []
    if hasattr(clf, 'feature_importances_'):
        fi = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        fi = numpy.sum(numpy.absolute(clf.coef_), axis=0)
    elif isinstance(clf, MANDALAClassifier):
        fi = clf.compute_feature_importances()
    elif hasattr(clf, 'estimators_') and hasattr(clf.estimators_[0], 'feature_importances_'):
        l_feat_imp = [sum(cls.feature_importances_ for cls in cls_list)
                      for cls_list in clf.estimators_]
        fi = numpy.array(l_feat_imp).sum(0)
    else:
        imps = permutation_importance(clf, x_test, y_test)
        fi = imps.importances_mean
    return fi / sum(fi)


def compute_permutation_feature_importances(clf, x_test, y_test, clf_names, feature_names):
    """
    Outputs feature ranking in building a Classifier
    :return: ndarray containing feature ranks
    """
    #imps = permutation_importance(clf, x_test, y_test)
    #i_values = imps.importances_mean/sum(imps.importances_mean)
    i_values = compute_feature_importances(clf, x_test, y_test)
    i_dict = {feature_names[i]: i_values[i] for i in range(len(feature_names))}
    out_dict = {'dataset_features': sum([i_dict[key] for key in i_dict if 'datasetfeature' in key])}
    for clf_i in range(0, len(clf_names)):
        out_dict['clf' + str(clf_i+1) + '_probas'] = \
            sum([i_dict[key] for key in i_dict if clf_names[clf_i] + '_prob' in key])
        for tag in ['label', 'maxp', 'ent']:
            out_dict['clf' + str(clf_i+1) + '_' + tag] = i_dict[clf_names[clf_i] + '_' + tag]
    return out_dict


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
    clf_metrics["adj"]["rec"] = sklearn.metrics.recall_score(test_y, adj_scores)
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


def make_dataset_dict(clf_names, dataset_df):
    dataset_dict = {}
    for clf_name in clf_names:
        clf_dict = {'label': dataset_df[clf_name + '_label'].to_numpy(),
                    'maxp': dataset_df[clf_name + '_maxp'].to_numpy(),
                    'ent': dataset_df[clf_name + '_ent'].to_numpy(),
                    'probas': numpy.asarray([dataset_df[x] for x in dataset_df.columns
                                             if x.startswith(clf_name) and '_prob' in x]).T}
        dataset_dict[clf_name] = clf_dict
    return dataset_dict


def make_ensemble_prediction(train_clf_dict, test_clf_dict, adj, train_labels=None, use_dataset=None):
    feat_imp = None
    meta_clf = None
    if hasattr(adj, 'fit') and callable(adj.fit):
        # Stacker
        meta_clf = copy.deepcopy(adj)
        train_p = []
        test_p = []
        column_names = []
        for clf in train_clf_dict:
            clf_data = [train_clf_dict[clf]['probas'], train_clf_dict[clf]['ent'].reshape((-1, 1)),
                        train_clf_dict[clf]['maxp'].reshape((-1, 1)), train_clf_dict[clf]['label'].reshape((-1, 1))]
            train_p.append(numpy.concatenate(clf_data, axis=1))
            clf_data = [test_clf_dict[clf]['probas'], test_clf_dict[clf]['ent'].reshape((-1, 1)),
                        test_clf_dict[clf]['maxp'].reshape((-1, 1)), test_clf_dict[clf]['label'].reshape((-1, 1))]
            test_p.append(numpy.concatenate(clf_data, axis=1))
        train_p = numpy.concatenate(train_p, axis=1)
        test_p = numpy.concatenate(test_p, axis=1)
        if use_dataset is not None:
            t_feat = use_dataset['train']
            train_p = numpy.concatenate([train_p, t_feat], axis=1)
            te_feat = use_dataset['test']
            test_p = numpy.concatenate([test_p, te_feat], axis=1)
        meta_clf.fit(train_p, train_labels)
        probas = meta_clf.predict_proba(test_p)
    elif adj == 'maxp':
        maxps = numpy.vstack([test_clf_dict[clf]['maxp'] for clf in test_clf_dict]).T
        best_clf = numpy.asarray(list(test_clf_dict.keys()))[numpy.argmax(maxps, axis=1)]
        probas = numpy.asarray([test_clf_dict[best_clf[i]]['probas'][i] for i in range(0, len(best_clf))])
    elif adj == 'best':
        best_clf = None
        for clf in train_clf_dict:
            acc = sklearn.metrics.balanced_accuracy_score(train_clf_dict[clf]['label'], train_labels)
            if best_clf is None or best_clf[1] < acc:
                best_clf = [clf, acc]
        probas = test_clf_dict[best_clf[0]]['probas']
    else:
        probas = None
        for clf in test_clf_dict:
            if probas is None:
                probas = copy.deepcopy(test_clf_dict[clf]['probas'])
            else:
                probas += test_clf_dict[clf]['probas']
        probas /= len(test_clf_dict)
    return {'probas': probas, 'label': numpy.argmax(probas, axis=1),
            'maxp': numpy.max(probas, axis=1), 'ent': entropy(probas)}, \
           [meta_clf, test_p] if meta_clf is not None else None


def compute_metrics(clf_name, clf_matrix, labels):
    metrics_dict = {'clf': clf_name}
    pred = clf_matrix['label']
    maxp = clf_matrix['maxp']
    metrics_dict['acc'] = sklearn.metrics.accuracy_score(pred, labels)
    metrics_dict['b_acc'] = sklearn.metrics.balanced_accuracy_score(pred, labels)
    metrics_dict['mcc'] = sklearn.metrics.matthews_corrcoef(pred, labels)
    metrics_dict['avg_conf'] = numpy.average(maxp)
    metrics_dict['avg_conf_hit'] = numpy.average(maxp[pred == labels])
    metrics_dict['avg_conf_misc'] = numpy.average(maxp[pred != labels])
    metrics_dict['test_size'] = len(pred)
    metrics_dict['n_classes'] = clf_matrix['probas'].shape[1]
    return metrics_dict
