import os

import keras
import tensorflow

from keras.applications import VGG16, ResNet50V2, MobileNet

from mandalalib.classifiers.PDIClassifier import PDIClassifier


def load_MIST_model():
    print(os.getcwd())
    print(tensorflow.__version__)
    mnist_model = tensorflow.keras.models.load_model('./mnist_model.keras')
    return mnist_model


class PDITLClassifier(PDIClassifier):
    """
    Wrapper for a keras sequential network
    """

    def __init__(self, n_classes, tl_tag='mobilenet', img_size=32, pdi_strategy='tsne',
                 epochs=50, bsize=1024, val_split=0.2, patience=3, verbose=2):
        if tl_tag == 'vgg':
            tl_model = VGG16(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
        elif tl_tag == 'resnet':
            tl_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
        elif tl_tag == 'mnist':
            img_size = 28
            tl_model = load_MIST_model()
        else:
            tl_model = MobileNet(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
        tl_model.trainable = True
        if tl_tag == 'mnist':
            is_rgb = False
            model = keras.Sequential(
                [
                    tl_model,
                    keras.layers.Dense(n_classes, activation='softmax')
                ]
            )
        else:
            is_rgb = True
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
            optimizer='adam', loss="categorical_crossentropy", metrics=[
                keras.metrics.CategoricalAccuracy(name='acc'),
                keras.metrics.AUC(name="auc")
            ]
        )
        PDIClassifier.__init__(self, n_classes, img_size, pdi_strategy, epochs, bsize,
                               val_split, patience, verbose, model, is_rgb)

    def classifier_name(self):
        return "PDITL(" + str(self.tl_tag) + "-" + \
               str(self.pdi_strategy) + "-" + str(self.img_size) + "-" + \
               str(self.epochs) + "-" + str(self.bsize) + "-" + str(self.val_split) + ")"
