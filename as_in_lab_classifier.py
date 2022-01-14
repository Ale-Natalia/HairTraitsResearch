from keras import Sequential
from keras.layers import MaxPooling2D, Dropout, LeakyReLU, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, Dense, Add, Input, GlobalAvgPool2D
from tensorflow.keras.models import Model


def resnet_block(input, filter_size=3, no_filters=16):
    layer1 = Conv2D(kernel_size=filter_size, filters=no_filters, padding="same")(input)
    layer2 = Conv2D(kernel_size=filter_size, filters=no_filters, padding="same")(layer1)
    return Add()([input, layer2])

def resize_block(input, filter_size=3, no_filters=16, initializer = None):
    layer1 = Conv2D(kernel_size=filter_size, filters=no_filters, strides=2, activation="relu", kernel_initializer=initializer)(input)
    layer2 = Conv2D(kernel_size=filter_size, filters=no_filters, padding="same", activation="relu", kernel_initializer=initializer)(layer1)
    resizedInput = Conv2D(kernel_size=3, filters=no_filters, strides=2, activation="relu", kernel_initializer=initializer)(input)
    return Add()([resizedInput, layer2])

def dropout(input, rate=0.2):
    droputLayer = Dropout(rate)(input)
    return Add()([input, droputLayer])

def build_mini_resnet1(input_size, num_classes):
    inputs = Input(shape=input_size)
    x = Conv2D(kernel_size=3, filters=16, strides=2)(inputs)

    x = resnet_block(x, no_filters=16)
    x = resnet_block(x, no_filters=16)

    x = resize_block(x, no_filters=32)
    x = resnet_block(x, no_filters=32)
    x = resnet_block(x, no_filters=32)
    #x = dropout(x)

    x = resize_block(x, no_filters=64)
    x = resnet_block(x, no_filters=64)
    x = resnet_block(x, no_filters=64)
    x = resnet_block(x, no_filters=64)

    x = resize_block(x, no_filters=128)
    x = resnet_block(x, no_filters=128)
    x = resnet_block(x, no_filters=128)
    x = resnet_block(x, no_filters=128)
    x = resnet_block(x, no_filters=128)
    x = resnet_block(x, no_filters=128)


    x = GlobalAvgPool2D()(x)
    x = Dense(num_classes, activation="relu")(x)
    return Model(inputs=inputs, outputs=x, name="mini_resnet")

"""
model = build_mini_resnet1((64, 64, 3), 12)
model.summary()
"""