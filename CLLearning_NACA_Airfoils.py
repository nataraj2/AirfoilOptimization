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
datafilename='./NACA_4digitGenerator/geometries/geomshift%04d.txt'%0
data = loadtxt(datafilename)
npoints = size(data[:,0])

TOTAL_SIZE=909

airfoil_data_full = zeros([TOTAL_SIZE,npoints,2,1])

for i in arange(0,TOTAL_SIZE,1):
	datafilename='./NACA_4digitGenerator/geometries/geomshift%04d.txt'%i
	data = loadtxt(datafilename)
	airfoil_data_full[i,:,:,0] = data

TRAINING_SIZE=909

airfoil_data = zeros([TRAINING_SIZE,npoints,2,1])

for i in arange(0,TRAINING_SIZE,1):
	datafilename='./NACA_4digitGenerator/geometries/geomshift%04d.txt'%i
	data = loadtxt(datafilename)
	airfoil_data[i,:,:,0] = data

#plt.plot(airfoil_data[100,:,0,0],airfoil_data[100,:,1,0])
#plt.axis('equal')
#plt.show()

# The data i.e. all values of x and y is already within 0 and 1

EPOCHS = 300

# Batch and shuffle the data
#train_dataset = tf.data.Dataset.from_tensor_slices(airfoil_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset = airfoil_data

def make_CLlearner_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same',
                                     input_shape=[npoints, 2, 1]))
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
	model.add(tf.keras.layers.Dense(128, input_shape=[npoints, 2, 1], activation='relu', kernel_initializer='he_uniform'))
	model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(layers.Flatten())
	model.add(tf.keras.layers.Dense(1))

	return model

CL = loadtxt('./ComputeCL/CoefficientOfLift.txt')

#CLlearner = make_CLlearner_model()
CLlearner = make_simpler_model()
#tf.keras.utils.plot_model(CLlearner, to_file='disc.png', show_shapes=True, show_layer_names=True)

loss_fn = tf.keras.losses.MeanSquaredError()

CLlearner.compile(optimizer='adam',
              loss=loss_fn)
optimizer = tf.keras.optimizers.Adam(1e-4)
CLlearner.fit(train_dataset, CL, epochs=1000)

checkpoint_dir = './Checkpoint_CL'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model=CLlearner,optimizer=optimizer)
checkpoint.save(file_prefix = checkpoint_prefix)

