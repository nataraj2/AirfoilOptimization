import tensorflow as tf
from tensorflow.keras import layers

from numpy import *
from scipy import *
import matplotlib.pyplot as plt
import os
import time
import sys
from random import randint

arguments = len(sys.argv)-1

pos = 1
while(arguments >= pos):
	arg = sys.argv[pos]
	is_restart = sys.argv[pos+1]
	print ("Parameter %i: %s %s"%(pos,sys.argv[pos],is_restart))
	pos = pos+2

#get one sample airfoil data and check for size
datafilename='./GenerateData/Airfoils/genairfoil%03d.txt'%0
data = loadtxt(datafilename)
npoints = size(data[:,0])


#get one sample airfoil data and check for size
datafilename='./GenerateData/NoiseInput/noiseinput%03d.txt'%0
data = loadtxt(datafilename)
noise_dim = size(data)

NOISE_DIM=noise_dim

TOTAL_SIZE=960

noise_input = zeros([TOTAL_SIZE,noise_dim])

for i in arange(0,TOTAL_SIZE,1):
	datafilename='./GenerateData/NoiseInput/noiseinput%03d.txt'%i
	data = loadtxt(datafilename)
	noise_input[i,:] = data

CL_predicted = zeros([noise_dim])

datafilename='CL_new_airfoils.txt'
data = loadtxt(datafilename)
CL_predicted = data


def make_CLlearner_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same',
                                     input_shape=[noise_dim,]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def make_simpler_model():
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(256, input_shape=[noise_dim,], activation='relu', kernel_initializer='he_uniform'))
	model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))
	model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))
	model.add(layers.Flatten())
	model.add(tf.keras.layers.Dense(1))

	return model


#CLlearner = make_CLlearner_model()
CLlearner = make_simpler_model()

loss_fn = tf.keras.losses.MeanSquaredError()

CLlearner.compile(optimizer='adam',
              loss=loss_fn)
optimizer = tf.keras.optimizers.Adam(1e-4)
CLlearner.fit(noise_input, CL_predicted, epochs=1000)

checkpoint_dir = './Checkpoint_CL_Noise'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model=CLlearner,optimizer=optimizer)
checkpoint.save(file_prefix = checkpoint_prefix)


