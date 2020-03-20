from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers import Conv2D
from keras.layers import LeakyReLU, ReLU
from keras.layers import Conv2DTranspose
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal


def generator_model_builder(input_shape, desired_output_shape):
    """
    :param input_shape: dimensions of the noise input to the generator
    :param desired_output_shape: dimensions of the final image generated by the generator (=discriminator input shape)
    """
    model = Sequential()
    assert desired_output_shape[0] % 4 == 0, "Error: architecture assumes input image size divides by 4."

    # weight initialization:
    init = RandomNormal(stddev=0.02)

    quarter_output_size = int(desired_output_shape[0] / 4)
    model.add(Dense(128 * quarter_output_size * quarter_output_size, input_dim=input_shape, kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((quarter_output_size, quarter_output_size, 128)))
    # up-sample to half output size:
    model.add(Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # up-sample to output size
    model.add(Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(filters=1, kernel_size=(7,7), activation='tanh', padding='same', kernel_initializer=init))

    return model