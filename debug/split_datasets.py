import os

from mandalalib.utils.MUtils import read_csv_dataset

LABEL_NAME = 'multilabel'
CSV_FOLDER = "datasets"
OUTPUT_FOLDER= "./split_multilabel_datasets"


if __name__ == '__main__':

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    for file in os.listdir(CSV_FOLDER):

        if file.endswith(".csv"):

            x_train, x_test, y_train, y_test, feature_list, att_perc = \
                read_csv_dataset(os.path.join(CSV_FOLDER, file))

            train_set = x_train
            train_set["label"] = y_train
            train_set.to_csv(os.path.join(OUTPUT_FOLDER, file.replace(".csv", "_train.csv")), index=False)

            test_set = x_test
            test_set["label"] = y_test
            test_set.to_csv(os.path.join(OUTPUT_FOLDER, file.replace(".csv", "_test.csv")), index=False)
