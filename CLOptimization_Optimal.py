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

checkpoint_dir = './Checkpoint_CL_Noise'
checkpoint = tf.train.Checkpoint(model=CLlearner)
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

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
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def generate_airfoil(iteration,seed,iter_vec,peri_vec,is_save):

	generate_and_save_images(generator,iteration,seed,iter_vec,peri_vec,is_save)
	
niter = 300
ntries = 50
optimal_no = 7

for tryno in arange(0,1,1):
	filename = 'seed_for_opt_%03d.txt'%optimal_no
	peri_vec = zeros(niter)

	seed_full = loadtxt(filename)
	seed =  zeros([1,noise_dim])
	seed[0,:] =  seed_full[0:noise_dim]
	seed1 = seed

	def generate_and_save_images(model,iteration,seed,iter_vec,peri_vec,is_save):
		fig, axes = plt.subplots(nrows=1, ncols=2,num=1)
		predictions1 = model(seed1, training=False)
		predictions = model(seed, training=False)
      		plt.subplot(1, 2, 1)
		plt.plot(predictions[0,:,0,0],predictions[0,:,1,0],linewidth=2)
		plt.plot(predictions1[0,:,0,0],predictions1[0,:,1,0],'r',label='Initial guess')
		plt.legend()
		plt.ylim([-0.15,0.15])
		plt.xlim([0.0,1.0])
		plt.axis('equal')

 	     	plt.subplot(1, 2, 2)
		plt.plot(iter_vec[0:iteration],peri_vec[0:iteration],'o',markersize=10,)
		plt.xlabel('Iteration')
		plt.ylabel('$C_L$')
		plt.suptitle('Iteration=%03d, CL=%f'%(iteration,peri_vec[iteration-1]),fontsize=20)
		fig.tight_layout(pad=2.0)
	
	 	if(is_save==1):
			imgfilename = './Images_Optimal/OptimalAirfoil%03d'%iteration
			plt.savefig(imgfilename)
		plt.draw()
		#plt.pause(0.001)
		plt.clf()

	# Do the optimization

	eps = 1e-6
	iter_vec = zeros(niter)

	def eps_array(indx):
		eps_array = zeros(noise_dim)
		eps_array[indx] = eps
	
		return eps_array

	f_x_temp = 0.0
	wait =80
	count=0
	for iteration in arange(0,niter,1):
		iter_vec[iteration] = iteration
		grad_f = zeros(noise_dim)
		#seed = tf.random.normal([1,noise_dim])+0.1*tf.random.normal([1,noise_dim])
        	f_x =  CLlearner.predict(seed)
		if(abs(f_x_temp-f_x)<=1e-6):
			count = count + 1
		if(abs(f_x_temp-f_x)<=1e-6 and count > wait):
			#generate_airfoil(iteration,seed,iter_vec,peri_vec,1)
			break

		f_x_temp = f_x
		peri_vec[iteration] = f_x
		generate_airfoil(iteration,seed,iter_vec,peri_vec,1)
		for ptindx in arange(0,noise_dim,1):
			seedpluseps = seed + eps_array(ptindx)
			f_xpluseps = CLlearner.predict(seedpluseps)
			grad_f[ptindx] = (f_xpluseps-f_x)/eps 	
		seed = seed + 0.05*grad_f
	







	
	
	
