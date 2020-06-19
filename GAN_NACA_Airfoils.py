import tensorflow as tf
from tensorflow.keras import layers

from numpy import *
from scipy import *
import matplotlib.pyplot as plt
import os
import time

import sys

arguments = len(sys.argv)-1

pos = 1
while(arguments >= pos):
	arg = sys.argv[pos]
	is_restart = sys.argv[pos+1]
	print ("Parameter %i: %s %s"%(pos,sys.argv[pos],is_restart))
	pos = pos+2
	arg = sys.argv[pos]
	is_training = sys.argv[pos+1]
	print ("Parameter %i: %s %s"%(pos,sys.argv[pos],is_training))
	pos = pos+2


TOTAL_SIZE=900

#get one sample airfoil data and check for size
datafilename='./NACA_4digitGenerator/geometries/geomshift%04d.txt'%0
data = loadtxt(datafilename)
npoints = size(data[:,0])
airfoil_data = zeros([TOTAL_SIZE,npoints,2,1])

for i in arange(0,TOTAL_SIZE,1):
	datafilename='./NACA_4digitGenerator/geometries/geomshift%04d.txt'%i
	data = loadtxt(datafilename)
	airfoil_data[i,:,:,0] = data

plt.plot(airfoil_data[100,:,0,0],airfoil_data[100,:,1,0])
plt.axis('equal')
#plt.show()

# The data i.e. all values of x and y is already within 0 and 1

	
BUFFER_SIZE = TOTAL_SIZE
BATCH_SIZE = 32
NOISE_DIM = 50

EPOCHS = 1600
noise_dim = NOISE_DIM
num_examples_to_generate = 16


# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(airfoil_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(npoints*2*32, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((npoints, 2, 32)))
    assert model.output_shape == (None, npoints, 2, 32) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(16, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, npoints, 2, 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(16, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, npoints, 2, 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, npoints, 2, 1)

    return model

generator = make_generator_model()
tf.keras.utils.plot_model(generator, to_file='gen.png', show_shapes=True, show_layer_names=True)

noise = tf.random.normal([1, NOISE_DIM])
generated_image = generator(noise, training=False)

#plt.plot(generated_image[0,:,0,0],generated_image[0,:,1,0])
#plt.axis('equal')
#plt.show()


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same',
                                     input_shape=[npoints, 2, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


discriminator = make_discriminator_model()
tf.keras.utils.plot_model(discriminator, to_file='disc.png', show_shapes=True, show_layer_names=True)
decision = discriminator(generated_image)
print (decision)

#exit()

# This method returns a helper function to compute cross entropy loss
#cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
cross_entropy = tf.keras.losses.MeanSquaredError()


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


checkpoint_dir = './Checkpoint_AirfoilGenerator'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
if(float(is_restart) == 1):
	status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim]) 

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
	for epoch in range(epochs):
		print epoch
		start = time.time()

		if(float(is_training) == 1):
    			for image_batch in dataset:
      				train_step(image_batch)

		generate_and_save_images(generator,
                	           epoch,
                        	   seed)

    # Save the model every 15 epochs
		if (epoch + 1) % 5 == 0:
			checkpoint.save(file_prefix = checkpoint_prefix)

 	fig = plt.figure(figsize=(4,4))
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)


  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      #plt.axis('equal')
      plt.plot(predictions[i,:,0,0],predictions[i,:,1,0],linewidth=2)
      #if(epoch > 500):
	#  plt.xlim([0.9,1.1])
	#  plt.ylim([-0.05,0.05])
      #elif(epoch > 20):
      plt.xlim([0.0,1.0])
      plt.ylim([-0.15,0.15])

      #plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      #plt.axis('off')
  plt.suptitle('Epoch = %03d'%epoch,fontsize=20)
  #plt.draw()
  #plt.pause(0.001)
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.clf()


train(train_dataset, EPOCHS)




