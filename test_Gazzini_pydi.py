"""
In this script we try pyDeepInsight pipeline on MechFailure dataset
We use a convolutional neural network.
See pyDeepInsight_performance.csv for details
It raises a Value Error when feature_extractor=tsne and we try to convert this tabular data into images matrices for CNN
"""
import numpy as np
import pandas as pd
import pyDeepInsight
from keras import models, layers
from keras.metrics import metrics
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('datasets/MechFailure_ElectricComponent_scikit.csv', sep=',', header=0)

X, y = data.iloc[:, 0:18], data.iloc[:, 18]

# Mapping labels to categorical binary value
y = y.map({"normal": 0, "Yes": 1})

# Train-Test splitting
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_test_unmodified = y_test

# Scaling values from x_train and x_test by using a class instance of MinMaxScaler (from sklearn.preprocessing)
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train, y_train)
x_test = min_max_scaler.transform(x_test)

# Usage of pyDeepInsight pipeline
image_transformer = pyDeepInsight.ImageTransformer(pixels=(70, 70))
image_transformer.fit(x_train)  # with fit method we have the possibility to show the
# scatter plot seeing the feature reduction, hull points, and minimum bounding rectangle
x_train = image_transformer.transform(x_train)
x_test = image_transformer.transform(x_test)

print(x_train.shape)    # (60800, 100, 100, 3)
print(x_test.shape)     # (15200, 100, 100, 3)

# One hot encoding of labels
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Now I have to build, as always, a convolutional neural network.
model = models.Sequential(layers=[
    layers.Conv2D(filters=8, kernel_size=3, padding='same', input_shape=(70, 70, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(70, activation='relu'),
    layers.Dense(35, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[metrics.AUC()]
)

# Train the model
model.fit(
    x=x_train,
    y=y_train,
    epochs=10,
    validation_split=0.2
)

# Evaluate the model
y_test_pred = model.predict(x=x_test)
y_test_pred = np.argmax(y_test_pred, axis=1)
acc = accuracy_score(y_true=y_test_unmodified, y_pred=y_test_pred)
mcc = matthews_corrcoef(y_true=y_test_unmodified, y_pred=y_test_pred)
matrix = confusion_matrix(y_true=y_test_unmodified, y_pred=y_test_pred)
print(f'Test Accuracy: {acc}')
print('--------------------------')
print(f'Matthews Correlation Coefficient: {mcc}')
print('--------------------------')
print(matrix)


