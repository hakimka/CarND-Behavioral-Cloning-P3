import cv2
import csv
import numpy as np
import sklearn

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images=[]
			measurements=[]
			for batch_sample in batch_samples:
				for i in range(3):	
					source_path = batch_sample[i]
					filename = source_path.split('\\')[-1]
					current_path='./data/IMG/'+filename	
					image = cv2.imread(current_path)	
					images.append(image)		
					if (i==0):
						measurement = float(batch_sample[3]) 
					else:
						measurement = float(batch_sample[3]) + 0.15*float((-1**(i+1)))
					measurements.append(measurement)
			aug_images=[]
			aug_measurements=[]	
			for image, measurement in zip (images, measurements):
				aug_images.append(image)
				aug_measurements.append(measurement)
				aug_images.append(cv2.flip(image,1))
				aug_measurements.append(measurement*-1.0)
	
	
			X_train = np.array(aug_images)
			y_train = np.array(aug_measurements)
			
			#yield (X_train, y_train)            
			yield sklearn.utils.shuffle(X_train, y_train)

samples  = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples .append(line)

		
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
		
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
	
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D

# convolution kernel size
nb_conv = 3

# number of convolutional filters to use
nb_filters = 32

nb_pool = 2 

model = Sequential()

# pre processing - normalize and crop
model.add(Lambda(lambda x: x/255.0-0.5, input_shape =(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

# conv layer #1
model.add(Convolution2D(24, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# conv layer #2
model.add(Convolution2D(36, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# conv layer #3
model.add(Convolution2D(48, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# conv layer #4
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.1))
#flatten and FC1
model.add(Flatten())
model.add(Dense(1164))
model.add(Activation('relu'))
#FC2
model.add(Dense(100))
model.add(Activation('relu'))
#FC3
model.add(Dense(50))
model.add(Activation('relu'))
#FC4
model.add(Dense(10))
model.add(Activation('relu'))
#FC5 - output
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), 
                           validation_data=validation_generator, 
                           nb_val_samples=len(validation_samples), nb_epoch=10)

#model.fit(X_train,y_train, validation_split=0.3, shuffle=True, nb_epoch=15)

model.save('model6.h5')
