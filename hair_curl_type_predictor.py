import cv2
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from data_read_curl_types import DataGenerator
from tensorflow import keras
from data_loading import DataGenerator


def square_image(image):
    width_pad = 0
    height_pad = 0
    if image.shape[0] > image.shape[1]:
        width_pad = (image.shape[0] - image.shape[1]) // 2
    else:
        height_pad = (image.shape[1] - image.shape[0]) // 2
    return np.pad(image, ((height_pad, height_pad), (width_pad, width_pad), (0, 0)), mode="edge")


def item_to_predict(path):
    """"
    Generates a batch of data
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = square_image(image)
    image = cv2.resize(image, (128, 128))
    return image


model_to_load_path = './resnet101_hair_classifier_model'
model = keras.models.load_model(model_to_load_path)
model.summary()
# use predict_classes for Sequential()
# prediction = model.predict_classes([item_to_predict('F:/UBB_Uni/an 3/Licenta/hair_pictures/hair_01_2c.jpg')])

# use predict for Model()
prediction = model.predict([item_to_predict('F:/UBB_Uni/an 3/Licenta/hair_pictures/hair_01_2c.jpg')])

print(prediction)
