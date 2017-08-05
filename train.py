import numpy as np
from sklearn.model_selection import train_test_split

from helper_functions import *
from models import * 

train_file = 'data/training.csv'

X_data, y_data = prepare_training_data(train_file)

X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size=0.10, random_state=42)


print("Shape of X_train: ", X_train.shape)
print("Shape of y_train: ", y_train.shape)

## Model paramters
batch_size = 128
epochs = 30
input_shape = (96, 96, 1)


model = get_model(input_shape)

## Train Classifier
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_valid, y_valid))

# Save model
model.save('model.h5')
