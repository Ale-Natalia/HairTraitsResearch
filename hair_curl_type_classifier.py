# TensorFlow and tf.keras
import math

import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from data_read_curl_types import DataGenerator
from model import resnet_block, build_mini_resnet, build_hair_classifier_model, build_hair_classifier_model_knn_101
from as_in_lab_classifier import build_mini_resnet1
from keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


print(tf.__version__)

"""
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
"""


def plot_evolution(fit_model):
    plt.subplot(1,2,1)
    plt.plot(fit_model.history['accuracy'], label='accuracy')
    plt.plot(fit_model.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='upper right')

    plt.subplot(1,2,2)
    plt.plot(fit_model.history['loss'], label='loss')
    plt.plot(fit_model.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.show()


def my_metric_fn(y_true, y_pred):
    print('true: {}, predicted: {}'.format(y_true, y_pred))
    prediction_difference = y_true - y_pred
    acceptablePredictions = np.count_nonzero(-1 <= prediction_difference <= 1)
    return acceptablePredictions // len(prediction_difference)  # Note the `axis=-1`


def train(batch_size, nr_epochs, learning_rate=None):
    db_dir = "F:/UBB_Uni/an 3/Licenta/hair_pictures"
    train_generator = DataGenerator(db_dir, batch_size, (128, 128, 3), 12)
    label_names = train_generator.class_names
    # assert len(label_names) == 12
    x, y = train_generator.get_data(db_dir)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=13,
                                                        shuffle=True)
    x_train = train_generator.get_images_from_paths_array(x_train)
    x_test = train_generator.get_images_from_paths_array(x_test)
    # batch_data, batch_labels = train_generator[0]

    # label_names = train_generator.class_names

    # model = build_mini_resnet((300, 300, 3), 12)
    # model = build_hair_classifier_model((128, 128, 3), 12)
    # model = build_hair_classifier_model_knn_101((128, 128, 3), 12)
    model = build_mini_resnet1((128, 128, 3), 12)
    if learning_rate is not None:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'],
                      run_eagerly=True)
    else:
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'], run_eagerly=True)
    history_logger = CSVLogger('as_in_lab_hair_classifier_model_relu.log', separator=',', append=False)
    fit_model = model.fit(x_train, y_train, epochs=nr_epochs, callbacks=[history_logger], validation_data=(x_test, y_test))

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    """
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    matrix = confusion_matrix(y_test, y_pred, labels=np.arange(12))
    probabilities = matrix.diagonal() / matrix.sum(axis=1)
    print(matrix)
    print(probabilities)
    """

    plot_evolution(fit_model)
    model.save('as_in_lab_hair_classifier_model_relu')



train(64, 50, 1e-4)

"""
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
"""