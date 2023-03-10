import keras

from keras.applications import VGG16, ResNet50V2, MobileNet

from mandalalib.classifiers.PDIClassifier import PDIClassifier


class PDITLClassifier(PDIClassifier):
    """
    Wrapper for a keras sequential network
    """

    def __init__(self, n_classes, tl_tag='mobilenet', img_size=70, pdi_strategy='tsne',
                 epochs=50, bsize=1024, val_split=0.2, verbose=2):
        self.img_t = None
        self.tl_tag = tl_tag
        self.img_size = img_size
        self.pdi_strategy = pdi_strategy
        self.epochs = epochs
        self.bsize = bsize
        self.verbose = verbose
        self.val_split = val_split
        self.norm_stats = {"train_avg": None, "train_std": None}
        if tl_tag == 'vgg':
            tl_model = VGG16(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
        elif tl_tag == 'resnet':
            tl_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
        else:
            tl_model = MobileNet(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
        tl_model.trainable = False
        model = keras.Sequential(
            [
                tl_model,
                keras.layers.Flatten(),
                keras.layers.Dense(200, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(img_size, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(int(img_size/2.0), activation='relu'),
                keras.layers.Dense(n_classes, activation='softmax')
            ]
        )
        model.compile(
            optimizer='adam', loss="binary_crossentropy", metrics=[
                keras.metrics.Accuracy(name="accuracy"),
                keras.metrics.AUC(name="auc")
            ]
        )
        PDIClassifier.__init__(self, n_classes, img_size, pdi_strategy, epochs, bsize, val_split, verbose)

    def classifier_name(self):
        return "PDITL_CNN(" + str(self.tl_tag) + "-" + \
               str(self.pdi_strategy) + "-" + str(self.img_size) + "-" + \
               str(self.epochs) + "-" + str(self.bsize) + "-" + str(self.val_split) + ")"
