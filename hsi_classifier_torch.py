import numpy as np
import pandas as pd
import os

from hsi_io import load_train,load_train_test,export_labels,save_train_history
from variables import *

from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torch_models import *

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
y_train_np = y_train.to_numpy()
X_test_np = X_test.to_numpy()
X_test_np = X_test_np.reshape((X_test_np.shape[0],X_test_np.shape[1],1))
y_test_np = y_test.to_numpy()
print('zerodata.shape = ', zerodata.shape)

class HSIDataset(Dataset):
     def __init__(self, X, y):
          self.X = X
          self.y = y

     def __len__(self):
          return self.X.shape[0]

     def __getitem__(self, idx):
          return self.X[idx], self.y[idx]

train_ds = HSIDataset(X_train_np,y_train_np)
test_ds = HSIDataset(X_test_np,y_test_np)

print()
print('train_ds.X shape = ',train_ds.X.shape)
print('test_ds.X shape = ',test_ds.X.shape)
print('train_ds.y shape = ',train_ds.y.shape)
print('test_ds.y shape = ',test_ds.y.shape)

# number of inputs
n1 = nbands
# number of outputs (classes) with additional zero class (not used for train)
num_classes = np.max(y_train) + 1

net = Net(n1,num_kernels,k1,k2,n4,num_classes)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=0.001)

batch_size = n_train_samples
train_loader = DataLoader(dataset=train_ds, batch_size=n_train_samples, shuffle=False)
test_loader = DataLoader(dataset=test_ds, batch_size=n_test_samples, shuffle=False)

# load model
if vars(args)["checkpoint"] is not None:
     model_to_load = vars(args)['checkpoint']
else:
     model_to_load = 'model_to_load'

if os.path.exists(model_to_load):
     net.load_state_dict(torch.load(model_to_load))
net.eval()


# Train the model
print('Start training:')

total_step = len(train_loader)
print('total_step=',total_step)
loss_list = []
acc_list = []
out_per_epoch = 3
for epoch in range(epochs):
     for i, (X,y) in enumerate(train_loader):
          outputs = net(X)
          y = y.squeeze().long()
          loss = criterion(outputs, y)
          loss_list.append(loss.item())
          # gradients set to zero
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
	      # Track the accuracy
          total = y.size(0)
          _, predicted = torch.max(outputs.data, 1)
          correct = (predicted == y).sum().item()
          acc_list.append(correct / total)
          
          if len(train_loader) == 1 or (i + 1) % (len(train_loader) // out_per_epoch) == 0:
               print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                     .format(epoch + 1, epochs, i + 1, total_step, loss.item(), (correct / total) * 100))

print('Finish training.')

# Test the model
print('Model evaluation:')

net.eval()
testList = []
with torch.no_grad():
     correct = 0
     total = 0
     for X, y in test_loader:
          outputs = net(X)
          _, predicted = torch.max(outputs.data, 1)
          y = y.squeeze().long()
          total += y.size(0)
          correct += (predicted == y).sum().item()
          testList.extend(predicted)

     print('Test Accuracy of the model: {} %'.format((correct / total) * 100))

MODEL_STORE_PATH = os.environ['PWD'] + '/output/'
# Save the model and plot
torch.save(net.state_dict(), MODEL_STORE_PATH + 'cnn_model_%d.ckpt' % epochs)

from bokeh.plotting import figure, output_file, save
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d

p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title='PyTorch CNN results')
p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
p.line(np.arange(len(loss_list)), loss_list)
p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
output_file(MODEL_STORE_PATH + 'myPlot_%d.html' % epochs)
save(p)

# from bokeh.resources import CDN
# from bokeh.embed import file_html
# html = file_html(p, CDN, "MYPlot")
# with open("myPlot1.html",'w') as f:
#      f.write(html)

trainList0 = []
trainList = []
for X, y in train_loader:
     outputs = net(X)
     _, predicted = torch.max(outputs.data, 1)
     for elem in predicted.flatten():
          trainList.append(elem.item())
     for elem in y:
          trainList0.append(elem.numpy())
# add 'labels' column
X_train['labels0'] = trainList0
X_train['labels'] = trainList

testList0 = []
testList = []
if X_test_np.shape[0] > 0:
     for X, y in test_loader:
          outputs = net(X)
          _, predicted = torch.max(outputs.data, 1)
          for elem in predicted.flatten():
               testList.append(elem.item())
          for elem in y:
               testList0.append(elem.numpy())
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
