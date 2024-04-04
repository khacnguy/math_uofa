import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np 
import pickle
import os
# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# normalize to range 0-1
	train_norm = train / 255.0
	test_norm = test / 255.0
	return train_norm, test_norm



# define cnn model
def define_model():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), padding = 'same', activation='relu', kernel_initializer='he_normal', input_shape=(14, 14, 1)))
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dense(10, activation='softmax'))
  opt = SGD(learning_rate=0.01, momentum=0.9)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  return model
# evaluate a model using k-fold cross-validation
def evaluate_model(trainX, testX, trainY, testY):
    histories = []
    accuracies = []
    kfold = KFold(6, shuffle=True, random_state=1)
    for train_ix, test_ix in kfold.split(trainX):
        model = define_model()
        history = model.fit(trainX[train_ix], trainY[train_ix], epochs=10, batch_size=32, validation_data=(trainX[test_ix], trainY[test_ix]), verbose=2)
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        histories.append(history)
        accuracies.append(acc)
    return histories, accuracies

def low_resolution(trainX, testX):
    avgpool = AveragePooling2D(pool_size = (2,2), strides = (2,2))
    trainX = avgpool(trainX)
    testX = avgpool(testX)
    return np.array(trainX), np.array(testX)

def mnist_train():    
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = low_resolution(trainX, testX)
    trainX, testX = prep_pixels(trainX, testX)
    return evaluate_model(trainX,  testX, trainY, testY)

def load_dataset_fashion():
	(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY
	# evaluate model
def define_model_fashion():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), padding = 'same', activation='relu', kernel_initializer='he_normal', input_shape=(28, 28, 1)))
  model.add(Dropout(0.2))
  model.add(BatchNormalization())
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
  model.add(Dropout(0.3))
  model.add(BatchNormalization())
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
  model.add(Dropout(0.4))
  model.add(BatchNormalization())
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dense(32, activation = 'relu'))
  model.add(Dense(10, activation='softmax'))
  # compile model
  opt = SGD(learning_rate=0.01, momentum=0.9)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  return model

# evaluate a model using k-fold cross-validation
def evaluate_model_fashion(trainX, testX, trainY, testY):
    histories = []
    accuracies = []
    kfold = KFold(6, shuffle=True, random_state=1)
    for train_ix, test_ix in kfold.split(trainX):
        model = define_model_fashion()
        #change epochs to 40 please
        history = model.fit(trainX[train_ix], trainY[train_ix], epochs=40, batch_size=32, validation_data=(trainX[test_ix], trainY[test_ix]), verbose=2)
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        histories.append(history)
        accuracies.append(acc)
    return histories, accuracies
def mnist_fashion_train():
    trainX, trainY, testX, testY = load_dataset_fashion()
    trainX, testX = prep_pixels(trainX, testX)
    return evaluate_model_fashion(trainX, testX, trainY, testY)
if __name__=="__main__":
    mnist_train()
    mnist_fashion_train()
    

