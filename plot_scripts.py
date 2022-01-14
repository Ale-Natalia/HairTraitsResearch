import math

import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

from data_read_curl_types import DataGenerator
from model import resnet_block, build_mini_resnet, build_hair_classifier_model, build_hair_classifier_model_knn_101
from as_in_lab_classifier import build_mini_resnet1
import pandas as pd


def plot_evolution(fit_model):
    plt.subplot(1,2,1)
    plt.plot(fit_model.history['accuracy'], label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='upper right')

    plt.subplot(1,2,2)
    plt.plot(fit_model.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

def plot_evolution_history(history):
    plt.subplot(1,2,1)
    plt.plot(history['accuracy'], label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='upper right')

    plt.subplot(1,2,2)
    plt.plot(history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.show()


model_to_load_path = './as_in_lab_hair_classifier_model'
model = tf.keras.models.load_model(model_to_load_path)
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


#history_log_data = pd.read_csv('as_in_lab_hair_classifier_model.log', sep=',', engine='python')
#plot_evolution_history(history_log_data)


