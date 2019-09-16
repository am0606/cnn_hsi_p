INIT_LR = 0.01
batch_size = 600
epochs = 1500
decay = INIT_LR/epochs
validation_split=0.1

from tensorflow.keras.optimizers import Adam,Adadelta,RMSprop,SGD
# opt = Adam(lr=INIT_LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay, amsgrad=False)
# opt = Adadelta(lr=INIT_LR, rho=0.95, epsilon=None, decay=decay)
# opt = RMSprop(INIT_LR, rho=0.9, epsilon=None, decay=0.0)
opt = SGD(lr=INIT_LR,momentum=0.9,decay=decay,nesterov=False)

nrows_image = 83
ncols_image = 86

# network parameters
# Wei Hu et al. Deep convolutional neural networks for hyperspectral image classification.
# J. Sensors 2015, article ID 258619 (2015).
 
# number of neurons in convolutional layer C1
num_kernels = 10 # 20
# С1 kernel size
k1 = 20 # int(n1 // 9)
k2 = 5 # int(n2 // n3), n2 = n1 - k1 + 1, n3 = 40
n4 = 100
