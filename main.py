
import warnings
warnings.filterwarnings("ignore")
from keras.datasets.mnist import load_data
import matplotlib
matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.optimizers import  RMSprop
import numpy as np
from discriminator import discriminator_model_builder, wasserstein_loss
from sklearn.preprocessing import MinMaxScaler
from generator import generator_model_builder
import pandas as pd
import time



def load_real_train_data():
    """
    :return: numpy array of real MNIST images
    """
    (trainX, trainy), (_, _) = load_data() # Load MNIST data
    # normalize:
    d0, d1, d2 = trainX.shape
    trainX = MinMaxScaler((-1,1)).fit_transform(trainX.reshape(d0, d1*d2)).reshape(d0,d1,d2)
    trainX = np.expand_dims(trainX.astype('float32'), axis=-1)

    return trainX


def gan_model_builder(generator_model, discriminator_model):
    """
    Builds and returns a Sequential model that includes the generator and the discriminator, sequentially.
    We will only update the generator's parameters in this Sequential model, as the Discriminator's params will be
    updates separately in the training loop.
    """
    discriminator_model.trainable = False # don't update the discriminator's params here
    gan_model = Sequential()
    gan_model.add(generator_model)
    gan_model.add(discriminator_model)
    opt = RMSprop(lr=0.00005)
    gan_model.compile(loss=wasserstein_loss, optimizer=opt)
    return gan_model

def generate_noise_input(input_dim, n_samples):
    return np.random.randn(input_dim * n_samples).reshape(n_samples, input_dim)

def generate_fake_sample(generator_model, input_dim, n_samples):
    X_input = generate_noise_input(input_dim, n_samples)
    fake_output = generator_model.predict(X_input)
    return fake_output

def plot_gan_losses(g_losses, d_losses_real, d_losses_fake, filename):
    n_losses = len(g_losses)
    plt.plot(range(n_losses), g_losses, label="Generator Loss")
    plt.plot(range(n_losses), d_losses_real, label="Discriminator Loss on Real Examples")
    plt.plot(range(n_losses), d_losses_fake, label="Discriminator Loss on Fake Examples")
    plt.legend()
    plt.savefig("./Data/generated_images/{}".format(filename))
    plt.close()

def train(generator_model, discriminator_model, gan_model, n_epochs, input_dim,
          train_data, batch_size, n_critic=5, save_generator_output_every_n_steps=50):
    batches_per_epoch = int(train_data.shape[0] / batch_size)
    half_batch_size = int(batch_size/2.0)
    n_steps = n_epochs * batches_per_epoch
    print("Begin training for {} epochs, {} batches of size {} per epoch. Total steps = {}".format(n_epochs, batches_per_epoch, batch_size, n_steps))
    g_losses, d_losses_real, d_losses_fake = [], [], []
    for step in range(n_steps):
            batch_time_start = time.time()
            discriminator_loss_real, discriminator_loss_fake = 0,0
            for _ in range(n_critic):
                # randomly select real examples:
                X_real = train_data[np.random.randint(0, train_data.shape[0], size=half_batch_size)] # half the batch will real, half fake
                y_real = -1*np.ones((X_real.shape[0], 1))
                # Train disc. for one batch on real examples:
                discriminator_loss_real += discriminator_model.train_on_batch(X_real, y_real)

                # get fake examples:
                X_fake = generate_fake_sample(generator_model, input_dim, n_samples=half_batch_size)
                y_fake = np.ones((X_fake.shape[0], 1))
                # Train disc. for one batch on fake examples:
                discriminator_loss_fake += discriminator_model.train_on_batch(X_fake, y_fake)

            discriminator_loss_real /= n_critic # take mean
            discriminator_loss_fake /= n_critic  # take mean
            # train the generator (via the "gan" model) for one batch:
            X_gan = generate_noise_input(input_dim, batch_size)
            y_gan = -1*np.ones((X_gan.shape[0],1)) # label as non-fake, so that disc. model will output the opposite
            generator_loss = gan_model.train_on_batch(X_gan, y_gan)

            took = time.time() - batch_time_start
            print("Step<%d> -- generator_loss = %.3f discriminator_loss_real = %.3f discriminator_loss_fake = %.3f [took %.1f sec]"\
                 % (step, generator_loss, discriminator_loss_real, discriminator_loss_fake, took))

            # Keep track of losses:
            g_losses.append(generator_loss)
            d_losses_fake.append(discriminator_loss_fake)
            d_losses_real.append(discriminator_loss_real)

            # generate example with generator every few steps and in the last one:
            if (step % save_generator_output_every_n_steps == 0) or (step == n_steps-1):
                # Generate & plot 9 examples with generator:
                n_examples = 9
                generated_examples = generate_fake_sample(generator_model, input_dim, n_samples=n_examples)
                generated_examples = (generated_examples + 1) / 2.0  # scale from [-1,1] to [0,1]

                for i in range(n_examples):
                    # define subplot
                    plt.subplot(3, 3, 1 + i)
                    plt.axis('off')
                    plt.imshow(generated_examples[i, :, :, 0], cmap='gray_r')
                plt.savefig("./Data/generated_images/step_{}".format(step))
                plt.close()

                # Save current generator model to file:
                generator_model.save("./Data/models/generator_step_{}.h5".format(step))

    # Save a plot of losses over the training procedure:
    plot_gan_losses(g_losses, d_losses_real, d_losses_fake, filename="Losses.png")
    # Save the loss history:
    loss_df = pd.DataFrame(np.array([g_losses, d_losses_real, d_losses_fake]).T,
                           columns=["g_losses", "d_losses_real", "d_losses_fake"])
    loss_df.to_csv("./Data/losses_history.csv")

NOISE_INPUT_DIM = 100

train_dataset = load_real_train_data()
img_size_v = train_dataset.shape[1]
img_size_h = train_dataset.shape[2]
DISCRIMINATOR_INPUT_SHAPE = (img_size_v, img_size_h, 1)
discriminator_model = discriminator_model_builder(DISCRIMINATOR_INPUT_SHAPE)
generator_model = generator_model_builder(input_shape=NOISE_INPUT_DIM, desired_output_shape=DISCRIMINATOR_INPUT_SHAPE)
gan_model = gan_model_builder(generator_model, discriminator_model)

print("Training on {} examples.".format(train_dataset.shape[0]))
train(generator_model, discriminator_model, gan_model, n_epochs=10, batch_size=64,
      input_dim=NOISE_INPUT_DIM, train_data=train_dataset, save_generator_output_every_n_steps=200)
