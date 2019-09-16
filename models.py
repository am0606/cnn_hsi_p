# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Activation, Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2

from variables import *

# model1
def model1(n1,num_classes):
    model = Sequential()
    model.add(Conv1D(kernel_size = (k1), filters = num_kernels, input_shape = (n1,1), activation="tanh", padding = 'valid'))
    model.add(MaxPooling1D(pool_size = (k2)))
    model.add(Flatten())
    model.add(Dense(n4, activation="tanh"))
    # model.add(Dense(n4, kernel_regularizer=l2(0.001),activation="tanh"))
    model.add(Dense(num_classes, activation="softmax"))
    return model

# model2 (w dropout)
def model2(n1,num_classes):
    model = Sequential()
    model.add(Conv1D(filters=num_kernels, kernel_size=(k1), input_shape=(n1,1), activation='tanh', padding = 'valid'))
    model.add(MaxPooling1D(pool_size=(k2)))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(n4, activation='tanh'))
    model.add(Dense(num_classes, activation='softmax'))
    model.build()
    return model

# model2 (w kernel_regularizer)
def model3(n1,num_classes):
    model = Sequential()
    model.add(Conv1D(kernel_size = (k1), filters = num_kernels, input_shape = (n1,1), activation="tanh", padding = 'valid'))
    model.add(MaxPooling1D(pool_size = (k2)))
    model.add(Flatten())
    model.add(Dense(n4, kernel_regularizer=l2(0.001),activation="tanh"))
    model.add(Dense(num_classes, activation="softmax"))
    return model
