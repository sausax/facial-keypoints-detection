import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import LearningRateScheduler


def get_model(input_shape):
	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
	model.add(Conv2D(3, kernel_size=(3, 3),
	                 activation='elu'))
	model.add(Conv2D(24, (3, 3), activation='elu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(128, (3, 3), activation='elu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(36, (3, 3), activation='elu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(48, (3, 3), activation='elu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), activation='elu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(1164, activation='elu'))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='elu'))
	model.add(Dropout(0.5))
	model.add(Dense(50, activation='elu'))
	model.add(Dropout(0.5))
	model.add(Dense(30, activation='tanh'))

	model.compile(loss='mean_squared_error', optimizer=Adam())

	return model