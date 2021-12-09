# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from data_read_curl_types import DataGenerator
from model import resnet_block, build_mini_resnet, build_hair_classifier_model, build_hair_classifier_model_knn_101

print(tf.__version__)

train_generator = DataGenerator("F:/UBB_Uni/an 3/Licenta/hair_pictures", 32, (128, 128, 3), 3)
label_names = train_generator.class_names
# assert len(label_names) == 37
batch_data, batch_labels = train_generator[0]
# label_names = train_generator.class_names

"""
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
"""

# model = build_mini_resnet((300, 300, 3), 12)
# model = build_hair_classifier_model((128, 128, 3), 12)
model = build_hair_classifier_model_knn_101((128, 128, 3), 12)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # for multi-class classification
              metrics=['accuracy'])
model.fit(batch_data, batch_labels, epochs=10, batch_size=32)
loss = model.evaluate(batch_data, batch_labels)
print(loss)


"""
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
"""