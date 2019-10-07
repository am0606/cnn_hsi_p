from __future__ import print_function
import numpy as np
import pandas as pd
import string
import os

from hsi_io import load_train,load_train_test,export_labels,save_train_history
from variables import *
from keras_models import *

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from kerastuner import RandomSearch, Hyperband, BayesianOptimization
from time import time

#debug
import sys
import pdb

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--tr", dest="train", required = True, default = "salinas.txt.zip", help="read train set from FILE", metavar="TRAIN_SET_FILE")
parser.add_argument("--te", dest="test", required = False, help="read test set from FILE", metavar="TEST_SET_FILE")
parser.add_argument("--trlabels", dest="train_labels", required = True, default = "salinas_labels.txt.zip", help="read train labels from FILE", metavar="TRAIN_LABELS_FILE")
parser.add_argument("--telabels", dest="test_labels", required = False, help="read test labels from FILE", metavar="TEST_LABELS_FILE")
parser.add_argument("-m", "--model", dest="checkpoint", required = False, help="read model with weights from FILE", metavar="MODEL_FILE")
parser.add_argument("--tuner", dest='tuner', action='store_true', help="activate tuner mode.")
parser.add_argument("--nosplit", dest='nosplit', action='store_true', help="use only train set without test")
args = parser.parse_args()

train_filename = vars(args)['train']
train_labels_filename = vars(args)['train_labels']
if vars(args)["test"] is not None:
     test_filename = vars(args)['test']
     test_labels_filename = vars(args)['test_labels']
     nbands,nrows,ncols,X_train,X_test,y_train,y_test,zerodata = load_train_test(train_filename, train_labels_filename, test_filename, test_labels_filename)
else:
     nbands,nrows,ncols,X_train,X_test,y_train,y_test,zerodata = load_train(train_filename, train_labels_filename, args.nosplit)

print('X_train shape = ', X_train.shape)
print('X_test shape = ', X_test.shape)

n_train_samples = X_train.shape[0]
print(n_train_samples, 'train samples')
n_test_samples = X_test.shape[0]
print(n_test_samples, 'test samples')

X_train_np = X_train.to_numpy()
X_train_np = X_train_np.reshape((X_train_np.shape[0],X_train_np.shape[1],1))
print('X_train_np.shape = ', X_train_np.shape)
X_test_np = X_test.to_numpy()
X_test_np = X_test_np.reshape((X_test_np.shape[0],X_test_np.shape[1],1))
print('X_test_np.shape = ', X_test_np.shape)
print('zerodata.shape = ', zerodata.shape)

# number of inputs
n1 = nbands
# number of outputs (classes) with additional zero class (not used for train)
num_classes = np.max(y_train) + 1

y_train = to_categorical(y_train.to_numpy(), num_classes)
y_test = to_categorical(y_test.to_numpy(), num_classes)

if args.tuner:
     # tune model1
     def build_model(hp):
          model = Sequential()
          model.add(Conv1D(kernel_size = (k1), filters = hp.Choice('filters', values=[5, 10, 15]),
                           input_shape = (n1,1), activation=hp.Choice('activation1',values=["tanh","relu"]), padding = 'valid'))
          model.add(MaxPooling1D(pool_size = (k2)))
          model.add(Flatten())
          model.add(Dense(units = hp.Int('units', min_value=80, max_value=120, step=5), activation=hp.Choice('activation2',values=["tanh","relu"])))
          model.add(Dense(num_classes, activation="softmax"))
          model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-1])), loss='categorical_crossentropy', metrics=['accuracy'])
          return model

     tuner = BayesianOptimization(
     build_model,                 
     objective='val_accuracy',    # metric to optimize
                                  # % of correct answers in validation set
     max_trials=5,
     directory='test_directory'   # directory to save trained models
     )

     tuner.search(X_train_np,            # data for training in numpy format
               y_train,                  # correct (desired) outputs
               batch_size=128,           
               epochs=40,
               validation_split=0.2,     # % of data to use for validation
               verbose=1
               )

     models = tuner.get_best_models(num_models=1)

     tuner.search_space_summary()

     for model in models:
          model.summary()
          model.evaluate(X_test_np, y_test, verbose=1)
          print()
     
     # pdb.set_trace()
     print(tuner.hyperparameters.values)
     tuner.results_summary()
     sys.exit()

model = model1(n1,num_classes)
print(model.summary())

# load model
if vars(args)["checkpoint"] is not None:
     model_to_load = vars(args)['checkpoint']
else:
     model_to_load = 'model_to_load'

if os.path.exists(model_to_load):
     print(model_to_load,' loaded')
     model.load_weights(model_to_load)

callbacks_list = [
    ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='accuracy', save_best_only=True, period = 100),
        EarlyStopping(monitor='accuracy', patience=100),
        TensorBoard(log_dir="logs/{}".format(time()))
]

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(X_train_np,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks_list,
                    validation_split=validation_split,
                    verbose=2)

save_train_history(history.history,'output/train_history.csv')
save_train_history(history.history,'output/train_history.mat')
save_train_history(history.history,'output/train_history.txt')

# save model with weights
model.save('output/my_model.h5')

score_train = model.evaluate(X_train_np, y_train, verbose=1)
print('Train loss:', score_train[0])
print('Train accuracy:', score_train[1])

if X_test_np.shape[0] > 0:
     score_test = model.evaluate(X_test_np, y_test, verbose=1)
     print('Test loss:', score_test[0])
     print('Test accuracy:', score_test[1])

preds_train = model.predict_proba(X_train_np)
trainList0 = []
trainList = []
for i,p in enumerate(preds_train):
     trainList0.append(np.argmax(y_train[i]))
     trainList.append(np.argmax(p))
# add 'labels' column
X_train['labels0'] = trainList0
X_train['labels'] = trainList

if X_test_np.shape[0] > 0:
     preds_test = model.predict_proba(X_test_np)
     testList0 = []
     testList = []
     for i,p in enumerate(preds_test):
          testList0.append(np.argmax(y_test[i]))
          testList.append(np.argmax(p))
     # add 'labels' column
     X_test['labels0'] = testList0
     X_test['labels'] = testList

if args.nosplit:
     alldata = pd.concat([X_test, zerodata])
else:
     alldata = pd.concat([X_train, X_test, zerodata])

# sort by index for correct representation as image
alldata.sort_index(inplace=True)

print()
print('X_train.shape = ',X_train.shape)
print('X_test.shape = ',X_test.shape)
print('zerodata.shape = ',zerodata.shape)
print('alldata.shape = ',alldata.shape)
print('num_classes = ',num_classes)

datalabels = alldata['labels'].to_numpy()
try:
     datalabels = datalabels.reshape(nrows_image,ncols_image)
     export_labels('datalabels.txt',datalabels)
except:
     print("Can't reshape array according to",(nrows_image,ncols_image))
