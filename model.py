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


if __name__ == '__main__':
    model = build_mini_resnet((100, 100, 3), 12)
    model.summary()
