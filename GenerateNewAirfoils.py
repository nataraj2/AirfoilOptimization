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

NOISE_DIM = 50

EPOCHS = 30
noise_dim = NOISE_DIM
num_examples_to_generate = 32

#get one sample airfoil data and check for size
datafilename='./NACA_4digitGenerator/geometries/geomshift%04d.txt'%0
data = loadtxt(datafilename)
npoints = size(data[:,0])

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

def train(epochs):
	for epoch in range(epochs):
		print epoch

		seed1 = tf.random.normal([num_examples_to_generate, noise_dim]) 
		seed2 = tf.random.normal([num_examples_to_generate, noise_dim])

		seed = (3*seed1+8*seed2)/11.0
		
		for i in arange(0,num_examples_to_generate,1):
			filename = './GenerateData/NoiseInput/noiseinput%03d.txt'%(i+epoch*num_examples_to_generate)
			file = open(filename,'w')
			for ptindx in arange(0,noise_dim,1):
				file.write("%f\n"%seed[i,ptindx])
			
			file.close()

		start = time.time()

		generate_and_save_images(generator,
                	           epoch,
                        	   seed)

    # Save the model every 15 epochs
		if (epoch + 1) % 5 == 0:
			checkpoint.save(file_prefix = checkpoint_prefix)

 	fig = plt.figure(figsize=(4,8))

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  for i in range(predictions.shape[0]):
      filename = './GenerateData/Airfoils/genairfoil%03d.txt'%(i+epoch*num_examples_to_generate)
      file = open(filename,'w')
      for ptindx in arange(0,npoints,1):	
      	file.write("%f %f \n" % (predictions[i,ptindx,0,0],predictions[i,ptindx,1,0]))
      file.close()
      plt.subplot(4, 8, i+1)
      #plt.axis('equal')
      plt.plot(predictions[i,:,0,0],predictions[i,:,1,0],linewidth=2)
     	
      plt.xlim([0,1])
      plt.ylim([-0.2,0.2])

  plt.suptitle('Epoch = %03d'%epoch,fontsize=20)
  #plt.draw()
  #plt.pause(0.001)
  plt.savefig('./Images/image_at_epoch_{:04d}.png'.format(epoch))
  #plt.show()
  plt.clf()

train(EPOCHS)




