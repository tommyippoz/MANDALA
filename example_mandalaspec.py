import numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

from mandalalib.EnsembleMetric import QStatMetric, SigmaMetric, CoupleDisagreementMetric, DisagreementMetric, \
    SharedFaultMetric
from mandalalib.MEnsemble import MEnsemble
from mandalalib.classifiers.MANDALAClassifier import TabNet
from mandalalib.classifiers.PDITLClassifier import PDITLClassifier
from mandalalib.utils.MUtils import read_csv_binary_dataset, current_ms, report

LABEL_NAME = 'multilabel'
CSV_FILE = "datasets/ADFANet_Meta.csv"

DIVERSITY_METRICS = [QStatMetric(), SigmaMetric(), CoupleDisagreementMetric(), DisagreementMetric(),
                     SharedFaultMetric(relative=False), SharedFaultMetric(relative=True)]

if __name__ == '__main__':
    """
    Example of the application of a MANDALAspec classifier to a dataset 
    (just edit the CSV_FILE path to load it)
    """

    # Reads CSV Dataset
    x_train, x_test, y_train, y_test, feature_list, att_perc = read_csv_binary_dataset(CSV_FILE, limit=10000)

    # MANDALAspec ensemble using
    #   - a RandomForest
    #   - TabNet
    #   - image classifier transferred from MobileNetV2 using 70x70x3 image
    #           and TSNE to convert tabular data into images
    #   plus LinearDiscriminantAnalysis as binary adjudicator
    m_ens = MEnsemble(models_folder="",
                      classifiers=[RandomForestClassifier(),
                                   TabNet(),
                                   PDITLClassifier(n_classes=len(numpy.unique(y_train)), img_size=70,
                                                   pdi_strategy='tsne', epochs=50, bsize=1024, val_split=0.3,
                                                   verbose=0)
                                   ],
                      diversity_metrics=DIVERSITY_METRICS,
                      bin_adj=LinearDiscriminantAnalysis(),
                      use_training=True)

    print("Ensemble " + m_ens.get_name())
    # Exercises the Ensemble
    start_time = current_ms()
    m_ens.fit(x_train, y_train, verbose=True)
    elapsed_tr = current_ms() - start_time
    ens_predictions, adj_data, clf_predictions, a, b = m_ens.predict(x_test)

    # Reports MEnsemble Stats
    metric_scores, clf_metrics = report(ens_predictions, clf_predictions, y_test,
                                        diversity_metrics=DIVERSITY_METRICS, verbose=False)
    print(m_ens.get_name() + " MCC: " + str(clf_metrics["adj"]["mcc"])
          + " Train Time: " + str(elapsed_tr) + " ms")
