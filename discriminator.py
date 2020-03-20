from keras.optimizers import Adam, RMSprop
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras import backend
from keras.constraints import Constraint
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal


def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)


class WeightClipper(Constraint):
    def __init__(self, c):
        self.c = c

    def __call__(self, p):
        return backend.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}

def discriminator_model_builder(input_shape, lr=0.00005):
    """
    :param input_shape: dimensions of the images the discriminator takes as input
    :param lr: learning rate for optimizer
    :return:
    """
    clip_constraint = WeightClipper(0.01)
    # weight initialization
    init = RandomNormal(stddev=0.02)

    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), padding='same', input_shape=input_shape,
                     kernel_constraint=clip_constraint, kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), padding='same', input_shape=input_shape,
                     kernel_constraint=clip_constraint, kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dense(1, activation="linear"))  # linear activation for WGAN architecture

    opt = RMSprop(lr=lr)
    model.compile(optimizer=opt,
                  loss=wasserstein_loss
                  )

    return model