import scipy
import sklearn


class DiversityMetric:
    """
   Abstract Class for ensemble metrics.
   """

    def get_name(self):
        """
        Returns the name of the metric (as string)
        """
        pass

    def compute_diversity(self, pred1, pred2, test_y):
        """
        Abstract method to compute diversity metric
        :param pred1: predictions of the first classifier
        :param pred2: predictions of the second classifier
        :param test_y: reference test labels
        :return: a float value
        """
        pass


class RSquaredMetric(DiversityMetric):
    """
    Ranker using the R-Squared Statistical Index from SciPy
    """

    def compute_diversity(self, pred1, pred2, test_y):
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pred1, pred2)
        return r_value ** 2

    def get_name(self):
        return "R-Squared"


class PearsonMetric(DiversityMetric):
    """
    Ranker using the Pearson Correlation Index from SciPy
    """

    def compute_diversity(self, pred1, pred2, test_y):
        return scipy.stats.pearsonr(pred1, pred2)[0]

    def get_name(self):
        return "Pearson"


class CosineSimilarityMetric(DiversityMetric):
    """
    Ranker using the Cosine Distance from SciPy
    """

    def compute_diversity(self, pred1, pred2, test_y):
        return 1 - scipy.spatial.distance.cosine(pred1, pred2)

    def get_name(self):
        return "Cosine"


class SpearmanMetric(DiversityMetric):
    """
    Ranker using the Spearman Statistical Index from SciPy
    """

    def compute_diversity(self, pred1, pred2, test_y):
        return scipy.stats.spearmanr(pred1, pred2)[0]

    def get_name(self):
        return "Spearman"


class ChiSquaredMetric(DiversityMetric):
    """
    Ranker using the Chi-Squared Statistical Index from Scikit-Learn
    """

    def compute_diversity(self, pred1, pred2, test_y):
        scaled_feat_values = sklearn.preprocessing.MinMaxScaler().fit_transform(pred1.reshape(-1, 1))
        return sklearn.feature_selection.chi2(scaled_feat_values, pred2)[0][0]

    def get_name(self):
        return "ChiSquared"


class MutualInfoMetric(DiversityMetric):
    """
    Ranker using the Mutual Information Index from Scikit-Learn
    """

    def compute_diversity(self, pred1, pred2, test_y):
        return sklearn.feature_selection.mutual_info_classif(pred1.reshape(-1, 1), pred2)[0]

    def get_name(self):
        return "MutualInfo"


class ANOVAMetric(DiversityMetric):
    """
    Ranker using the ANOVA R from Scikit-Learn
    """

    def compute_diversity(self, pred1, pred2, test_y):
        return sklearn.feature_selection.f_classif(pred1.reshape(-1, 1), pred2)[0]

    def get_name(self):
        return "ANOVAF"


class QStatDiversity(DiversityMetric):
    """
    Ranker using the QStat Diversity
    """

    def compute_diversity(self, pred1, pred2, test_y):
        n11 = sum((pred1 == test_y) * (pred2 == test_y))
        n10 = sum((pred1 == test_y) * (pred2 != test_y))
        n01 = sum((pred1 != test_y) * (pred2 == test_y))
        n00 = sum((pred1 != test_y) * (pred2 != test_y))
        return (n11*n00 - n01*n10)/(n11*n00 + n01*n10)

    def get_name(self):
        return "QStat"
