# Generating MNIST Images With Convolutional Wasserstein-GAN 
An implementation of a Convolutional Generative Adversrial Network with Wasserstein loss, that successfully learns to generate realistic hand-written digits as seen in the [MNSIT data set](https://en.wikipedia.org/wiki/MNIST_database).
Implementation is based on basic Keras layers with tensorflow backend, in python 3.7.
This project was inspired by https://arxiv.org/pdf/1905.02417.pdf and [this blog post]( https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/).

## Training Process
The network was trained on an AWS `p3.2xlarge` GPU instance, taking about 15 minutes to run for 10 epochs on the whole MNIST dataset (60K images).
The network was trained with batch size of 64, resulting in ~9300 training steps. While the final network generates good images, decent ones are generated as soon as step 3000.
The Wasserstein-loss values of the generator and the discriminator (or critic, as this is a Wasserstein architecture) are recorded throughout the training process (discriminator loss is split to its loss on the real images and on the “fake” ones):
![loss_graph]

## Examples of Generated Images
At start of training:

![step_0]

After 1000 training steps:

![step_1000]

After 3000 training steps:

![step_3000]

After 5000 training steps:

![step_5000]

After 9000 training steps:

![step_9000]

## Trained Models
Trained generator model can be found [here](). The model was dumped to file every 1000 steps.

## Installing
see `requirements.txt` file

[loss_graph]: https://github.com/yoav1412/gan-mnist-image-generation/blob/master/Data/generated_images/Losses.png
[step_0]: https://github.com/yoav1412/gan-mnist-image-generation/blob/master/Data/generated_images/step_0.png
[step_1000]: https://github.com/yoav1412/gan-mnist-image-generation/blob/master/Data/generated_images/step_1000.png
[step_3000]: https://github.com/yoav1412/gan-mnist-image-generation/blob/master/Data/generated_images/step_3000.png
[step_5000]: https://github.com/yoav1412/gan-mnist-image-generation/blob/master/Data/generated_images/step_5000.png
[step_9000]: https://github.com/yoav1412/gan-mnist-image-generation/blob/master/Data/generated_images/step_9000.png
