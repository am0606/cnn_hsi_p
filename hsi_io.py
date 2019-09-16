import numpy as np
import string
import zipfile
import os

import torch
from torch.utils.data import Dataset
from scipy.io import loadmat,savemat
import pandas as pd

#debug
import sys
import pdb

def import_data(filename):
     """import data according to the first line: nbands nrows ncols"""
     if extension(filename) == 'mat':
          bands_mat = loadmat(filename)
          bands_matrix = bands_mat.popitem()[1]
          
          nbands = bands_matrix.shape[0]
          nrows = bands_matrix.shape[1]
          # pdb.set_trace()
          if len(bands_matrix.shape) == 3:
               ncols = bands_matrix.shape[2]
          elif len(bands_matrix.shape) == 2:
               ncols = 1
          else:
               print('Incorrectbands_matrix.shape')

          # check scaling to [-1;1]
          eps = 0.1          
          if abs(np.max(bands_matrix) - np.min(bands_matrix)) > eps:
               bands_matrix_new = -1 + 2*(bands_matrix - np.min(bands_matrix))/(np.max(bands_matrix) - np.min(bands_matrix))
          else:
               bands_matrix_new = bands_matrix

          # construct 2D array
          data = np.empty([nrows*ncols,nbands])
          if len(bands_matrix.shape) > 2:
               for i in range(nrows):
                    for j in range(ncols):
                         data[i*ncols+j,:] = bands_matrix_new[:,i,j]
          else:
               for i in range(nrows):
                    data[i,:] = bands_matrix_new[:,i]
     else:
          try:
               archive = zipfile.ZipFile(filename, 'r')
               f = archive.open(filename[:-4], 'r')
          except:
               f = open(filename, 'r')
          words=[]
          for line in f.readlines():
               for word in line.split():
                    words.append(word)
          nbands = np.int_(words[0])
          nrows = np.int_(words[1])
          ncols = np.int_(words[2])
          
          # training set: number of pixels with nbands -> number of inputs
          offset = 3 # offset is due to header (nbands,nrows,ncols) in words
          data = np.empty([nrows*ncols,nbands])
          if ncols > 1:
               for row in range(nrows):
                    for col in range(ncols):
                         for i in range(nbands):
                              gidx = (col*nrows + row)*nbands + i + offset
                              data[row*ncols+col,i] = np.float32(words[gidx])
          else:
               for row in range(nrows):
                    for i in range(nbands):
                         gidx = i*nrows + row + offset
                         data[row,i] = np.float32(words[gidx])

          f.close()
          # pdb.set_trace()
     return nbands,nrows,ncols,data

def import_labels(filename):
     """import data according to the first line: nrows ncols"""
     if extension(filename) == 'mat':
          labels_mat = loadmat(filename)
          labels_matrix = labels_mat.popitem()[1]
          nrows = labels_matrix.shape[0]
          ncols = labels_matrix.shape[1]
          labels = labels_matrix.reshape(nrows*ncols)
     else:
          try:
               archive = zipfile.ZipFile(filename, 'r')
               f = archive.open(filename[:-4], 'r')
          except:
               f = open(filename, 'r')
          words=[]
          for line in f.readlines():
               for word in line.split():
                    words.append(word)
          nrows = np.int_(words[0])
          ncols = np.int_(words[1])
          labels = np.zeros(nrows*ncols)
          offset = 2 # offset is due to header (nrows,ncols) in words
          for row in range(nrows):
               for col in range(ncols):
                    labels[row+col*nrows] = np.float32(words[row+col*nrows+offset])

          f.close()
     return nrows,ncols,labels

def export_data(filename, X, nbands):
     """export data according to the first line: nbands nrows ncols"""
     f = open(filename, 'a')
     n_pixels = len(X)
     f.write('{} {} 1\n'.format(nbands,n_pixels))
     for i in range(n_pixels):
          a = X[i]
          np.savetxt(f,a,fmt="%.15f")
     f.close()


def export_labels(filename, y):
     """import data according to the first line: nbands nrows ncols"""
     f = open(filename, 'w')
     n_rows = y.shape[0]
     try:
          n_cols = y.shape[1]
     except:
          n_cols = 1
     f.write('{} {}\n'.format(n_rows,n_cols))
     
     def end_of_line(idx,n_cols):
          if idx == n_cols-1:
               f.write('\n')
               idx = 0
          else:
               idx = idx + 1
          return idx

     idx = 0
     for elem in y:
          try:
               for e in elem:
                    f.write(' {:.0f}'.format(e))
                    idx = end_of_line(idx,n_cols)
          except:
               f.write(' {:.0f}'.format(elem))
               idx = end_of_line(idx,n_cols)
     f.close()

def extension(filename):
     return filename[-3:]

def load_zerodata(filename):
     try:
          zerodata = pd.read_csv(filename, delimiter = "\t")
     except:
          print("No '%s' file exists!" % filename)
     return zerodata

def save_zerodata(zerodata, filename):
     """save data from zero class"""
     print('zerodata.shape = ',zerodata.shape)
     if zerodata.shape[0] > 0:
          zerodata.to_csv(filename, sep='\t')
     else:
          print('zerodata.shape = ', zerodata.shape)
          print("No 'zerodata' to save!")

# build test set from train
def load_train(bandsfilename,labelsfilename,nosplit=False):
     """import data with splitting train and test sets and zero classes filtering"""
     [nbands,nrows,ncols,data] = import_data(bandsfilename)
     [nrows,ncols,labels] = import_labels(labelsfilename)

     # create dataframe
     alldata = pd.DataFrame(data)
     alldata['labels'] = labels
     alldata['labels'] = alldata['labels'].astype('int64')
    
     # get data according to '0' classes     
     zerodata = alldata[alldata.labels == 0]

     # # debug
     # zerodata_filename = 'zerodata.csv'
     # save_zerodata(zerodata,zerodata_filename)
     
     # filter zero classes
     alldata = alldata[alldata.labels != 0]

     if nosplit:
          X_train = alldata.iloc[:,:-1]
          y_train = alldata.iloc[:,-1]
          X_test = X_train
          y_test = y_train
     else:
          from sklearn.model_selection import train_test_split
          X_train, X_test, y_train, y_test = train_test_split(alldata.iloc[:,:-1], alldata.iloc[:,-1], test_size=0.2, random_state=42)
     
     # export train and test data and labels 
     offset = -4 #.txt
     if bandsfilename[-3:] == 'zip':
          offset = -8 #.zip
     train_data_fname = 'output/' + bandsfilename[:offset] + '_train.txt'
     train_labels_fname = 'output/' + labelsfilename[:offset] + '_train.txt'
     test_data_fname = 'output/' + bandsfilename[:offset] + '_test.txt'
     test_labels_fname = 'output/' + labelsfilename[:offset] + '_test.txt'
     if os.path.exists(train_data_fname):
          os.remove(train_data_fname)
     if os.path.exists(train_labels_fname):
          os.remove(train_labels_fname)
     if os.path.exists(test_data_fname):
          os.remove(test_data_fname)
     if os.path.exists(test_labels_fname):
          os.remove(test_labels_fname)

     export_data(train_data_fname, X_train.to_numpy(), nbands)
     export_labels(train_labels_fname, y_train.to_numpy())
     export_data(test_data_fname, X_test.to_numpy(), nbands)
     export_labels(test_labels_fname, y_test.to_numpy())
     return nbands,nrows,ncols,X_train,X_test,y_train,y_test,zerodata

def load_train_test(train_filename,train_labels_filename,test_filename,test_labels_filename):
     """import preprocessed train and test data"""
     [nbands_train,nrows_train,ncols_train,X_train] = import_data(train_filename)
     [nrows_train,ncols_train,y_train] = import_labels(train_labels_filename)

     # create dataframe train
     alldata_train = pd.DataFrame(X_train.reshape(X_train.shape[0],X_train.shape[1]))
     alldata_train['labels'] = y_train
     alldata_train['labels'] = alldata_train['labels'].astype('int64')
     # # filter zero classes
     zerodata_train = alldata_train[alldata_train.labels == 0]
     alldata_train = alldata_train[alldata_train.labels != 0]
     
     [nbands_test,nrows_test,ncols_test,X_test] = import_data(test_filename)
     [nrows_test,ncols_test,y_test] = import_labels(test_labels_filename)

     # create dataframe test
     alldata_test = pd.DataFrame(X_test.reshape(X_test.shape[0],X_test.shape[1]))
     alldata_test['labels'] = y_test
     alldata_test['labels'] = alldata_test['labels'].astype('int64')
     # filter zero classes
     zerodata_test = alldata_test[alldata_test.labels == 0]
     alldata_test = alldata_test[alldata_test.labels != 0]

     zerodata = pd.concat([zerodata_train, zerodata_test])

     # zerodata_filename = 'zerodata.csv'
     # save_zerodata(zerodata,zerodata_filename)

     X_train = alldata_train.iloc[:,:-1]
     y_train = alldata_train.iloc[:,-1]
     X_test = alldata_test.iloc[:,:-1]
     y_test = alldata_test.iloc[:,-1]
          
     if nbands_train == nbands_test:
          nbands = nbands_train
     else:
          print("Error in data files")
          sys.exit(1)
     return nbands,nrows_train,ncols_train,X_train,X_test,y_train,y_test,zerodata

def debug_print(filename,arr):
     file=open(filename,'w')
     for elem in arr:
          file.write('{}\n'.format(elem))
     file.close()

def save_train_history(hist_dict, filename):
     """ save train history"""
     hist_df = pd.DataFrame(hist_dict)
     if extension(filename) == 'csv':
          with open(filename, 'w') as f:
               hist_df.to_csv(f, sep='\t')
     elif extension(filename) == 'mat':
          savemat(filename, {
               'loss': hist_df['loss'].to_numpy(),
               'accuracy': hist_df['accuracy'].to_numpy(),
               'val_loss': hist_df['val_loss'].to_numpy(),
               'val_accuracy': hist_df['val_accuracy'].to_numpy()
          })
     elif extension(filename) == 'txt':
          with open(filename, 'w') as f:
               f.write("loss\taccuracy\tval_loss\tval_accuracy\n")
               np.savetxt(f, hist_df.to_numpy(), delimiter='\t')

