### Udacity Self Driving Car Engineer Nanodegree ###
####################################################
### Project 003 : Behavioral Cloning ###############
####################################################

import os
import csv

# Reading CSV files for data corresponding to 
# Reverse on Track 1
# Section of Track 2
# Reinforcement of unbounded turns and adding good driving behaviours 
# Correction for clean turning and angle of approach for sharp turns

samples = []
for i in range(7):
    with open('../data/data/driving_log_{}.csv'.format(i)) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2) #Validation set size : 20%

CAMERAS = 3 	#Left, Right, Centre

print("No. of training samples   : ", CAMERAS * len(train_samples))
print("No. of validation samples : ", CAMERAS * len(validation_samples))

import cv2
import numpy as np
import sklearn
from random import shuffle

#Correction for camera offset 
correction = 0.2 

def generator(samples, batch_size=8):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):								#Using left right and centre data for training
                    name = './../data/data/IMG/'+batch_sample[i].split('/')[-1]
                    center_image = cv2.imread(name)
                    center_angle = float(batch_sample[3])  + ( i - 3*(i//2) ) * correction	#Correcting for camera offset from centre 
                    images.append(center_image)
                    angles.append(center_angle)
#                    images.append(cv2.flip(center_image,1))					#Adding flipped images to training set
#                    angles.append(center_angle*(-1.0))
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=8)		#Actual Batch Size : CAMERA * Batch Size (L,R,C)
validation_generator = generator(validation_samples, batch_size=8)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#Create Keras Sequential Model
model = Sequential()

#Normalize Image RGB values around 0
model.add(Lambda(lambda x: (x / 255.0) - 0.5 , input_shape=(160, 320, 3)))

#Cropping image to generate a ROI and remove unnecessary data (TOP : 70px BOTTOM: 25px LEFT : 0px RIGHT : 0px
model.add(Cropping2D(cropping=((70,25),(0,0))))

#Neural Network based on "End to End Learning for Self Driving Cars", Mariusz Bojarski Et Al NVIDIA Corporation
#Deep Neural Network with 5 convolution layers
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))	#1st Convolution with 24 channels and 5x5 kernel
model.add(Dropout(0.2))							#Dropout Layer for Regularization
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))	#2nd Convolution with 36 channels and 5x5 kernel
model.add(Dropout(0.2))							#Dropout for Regularization
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))	#3rd Convolution with 48 channels and 5x5 kernel
model.add(Convolution2D(64,3,3,activation="relu"))			#1st Convolution with 64 channels and 3x3 kernel
model.add(Convolution2D(64,3,3,activation="relu"))			#1st Convolution with 64 channels and 3x3 kernel

#Creating fully connected layers of width 100, 50, 10
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1)) #Steering turn angle value

model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) #Using ADAM optimizer on Mean Squared Error Loss 
model.summary()

# Epochs : 3
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*CAMERAS, validation_data=validation_generator, nb_val_samples=len(validation_samples)*CAMERAS, nb_epoch=3)

model.save('model.h5')
