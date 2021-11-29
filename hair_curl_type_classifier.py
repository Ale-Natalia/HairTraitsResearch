# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from data_read_curl_types import DataGenerator
from model import resnet_block, build_mini_resnet


print(tf.__version__)

train_generator = DataGenerator("F:/UBB_Uni/an 3/Licenta/hair_pictures", 32, (200, 200, 3), 3)
label_names = train_generator.class_names

"""
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
"""

model = build_mini_resnet((100, 100, 3), 12)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_generator.data, train_generator.labels, epochs=10)

"""
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
"""