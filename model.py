from keras import Sequential
from keras.layers import MaxPooling2D, Dropout, LeakyReLU, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, Dense, Add, Input, GlobalAvgPool2D
from tensorflow.keras.models import Model
from data_read_curl_types import *


def resnet_block(input, filter_size=3, no_filters=16):
    layer1 = Conv2D(kernel_size=filter_size, filters=no_filters, padding="same")(input)
    layer2 = Conv2D(kernel_size=filter_size, filters=no_filters, padding="same")(layer1)
    return Add()([input, layer2])


def build_mini_resnet(input_size, num_classes):
    inputs = Input(shape=input_size)
    x = Conv2D(kernel_size=3, filters=16, strides=2)(inputs)
    x = resnet_block(x)
    x = resnet_block(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(num_classes)(x)
    return Model(inputs=inputs, outputs=x, name="mini_resnet")

def build_hair_classifier_model(input_size, num_classes):
    model = Sequential()

    # layer-1
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', input_shape=input_size, padding='same', strides=2))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', strides=2))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # layer-2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', strides=2))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    # layer-3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', strides=2))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.30))

    # layer-4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', strides=2))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.30))

    # layer-5
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', strides=2))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # layer-6
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', strides=1))
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', strides=1))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # layer-7
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', strides=1))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', strides=1))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # Last Layer
    model.add(Flatten())
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def build_hair_classifier_model_knn_101(input_size, num_classes):
    return tf.keras.applications.resnet.ResNet101(
        include_top=False, weights='imagenet', input_tensor=None,
        input_shape=input_size, pooling='avg', classes=num_classes
    )


def build_mini_resnet_clothes(input_size, num_classes):
    inputs = Input(shape=input_size)
    x = Conv2D(kernel_size=3, filters=16, strides=2)(inputs)
    x = resnet_block(x)
    x = resnet_block(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(num_classes)(x)
    return Model(inputs=inputs, outputs=x, name="mini_resnet")


if __name__ == '__main__':
    model = build_mini_resnet((100, 100, 3), 12)
    model.summary()
