# Simple CNN model in TensorFlow roughly based on https://keras.io/examples/cifar10_cnn/
import gin

from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, \
    Input, Activation, Flatten, Dropout
from tensorflow.keras.models import Model

from src import DATA_FORMAT


@gin.configurable
def SimpleCNN(input_shape=(3, 32, 32), dropout=0.0, n_filters=32, activation="relu",
                 n_dense=128, kernel_size=3, n1=1, n2=1, n_classes=10, bn=False):
    inputs = Input(shape=input_shape)
    x = inputs

    for id in range(n1):
        prefix_column = str(id) if id > 0 else ""
        x = Conv2D(n_filters, (kernel_size, kernel_size), padding='same', data_format=DATA_FORMAT,
                   name=prefix_column + "conv1")(x)
        if bn:
            x = BatchNormalization(axis=1, name=prefix_column + "bn1")(x)
        x = Activation(activation, name=prefix_column + "act_1")(x)
        x = Conv2D(n_filters, (kernel_size, kernel_size), data_format=DATA_FORMAT, padding='same',
                   name=prefix_column + "conv2")(x)
        if bn:
            x = BatchNormalization(axis=1, name=prefix_column + "bn2")(x)
        x = Activation(activation, name=prefix_column + "act_2")(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format=DATA_FORMAT)(x)

    for id in range(n2):
        prefix_column = str(id) if id > 0 else ""
        x = Conv2D(n_filters * 2, (kernel_size, kernel_size), data_format=DATA_FORMAT,
                   padding='same', name=prefix_column + "conv3")(x)
        if bn:
            x = BatchNormalization(axis=1, name=prefix_column + "bn3")(x)
        x = Activation(activation, name=prefix_column + "act_3")(x)
        x = Conv2D(n_filters * 2, (kernel_size, kernel_size), data_format=DATA_FORMAT, padding='same',
                   name=prefix_column + "conv4")(x)
        if bn:
            x = BatchNormalization(axis=1, name=prefix_column + "bn4")(x)
        x = Activation(activation, name=prefix_column + "act_4")(x)

    x = MaxPooling2D(pool_size=(2, 2), data_format=DATA_FORMAT)(x)
    x = Flatten()(x)

    x = Dense(n_dense, name="dense2")(x)
    if bn:
        x = BatchNormalization(name="bn5")(x)
    x = Activation(activation, name="act_5")(x)  # Post act
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Dense(n_classes, activation="linear", name="pre_softmax")(x)
    x = Activation(activation="softmax", name="post_softmax")(x)

    model = Model(inputs=[inputs], outputs=[x])

    return model


if __name__ == "__main__":
    model = SimpleCNN_tf()
