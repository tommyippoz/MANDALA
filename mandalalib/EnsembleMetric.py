from mandalalib.DiversityMetric import QStatDiversity


class EnsembleMetric:
    """
   Abstract Class for ensemble metrics.
   """

    def get_name(self):
        """
        Returns the name of the metric (as string)
        """
        pass

    def compute_diversity(self, clf_predictions, test_y):
        """
        Abstract method to compute diversity metric
        :param clf_predictions: predictions of clfs in the ensemble
        :param test_y: reference test labels
        :return: a float value
        """
        pass


class QStatMetric(EnsembleMetric):
    """
    Computes uncertainty via Maximum probability assigned to a class for a given data point.
    Higher probability means high uncertainty / confidence
    """

    def __init__(self):
        """
        Constructor Method
        """
        return

    def compute_diversity(self, clf_predictions, test_y):
        """
        Abstract method to compute diversity metric
        :param clf_predictions: predictions of clfs in the ensemble
        :param test_y: reference test labels
        :return: a float value
        """
        n_clfs = clf_predictions.shape[1]
        qd = QStatDiversity()
        diversities = [qd.compute_diversity(clf_predictions[:, i], clf_predictions[:, j], test_y)
                       for i in range(0, n_clfs-1) for j in range(i+1, n_clfs)]
        return 2*sum(diversities)/(n_clfs*(n_clfs-1))

    def get_name(self):
        return 'QStat'
