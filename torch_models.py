import torch
import torch.nn as nn
from torch.nn import init
torch.set_default_tensor_type('torch.DoubleTensor')

class Net(nn.Module):
     @staticmethod
     def weight_init(m):
          if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
               init.uniform_(m.weight, -0.05, 0.05)
               init.zeros_(m.bias)

     def _get_final_flattened_size(self):
          with torch.no_grad():
               x = torch.zeros(1, 1, self.input_channels)
               x = self.pool(self.conv(x))
          return x.numel() 

     def __init__(self, input_channels, n_kernels, kernel_size, pool_size, n4, n_classes):
          super(Net, self).__init__()
          # [The first hidden convolution layer C1 filters the input_channels x 1 input data with 20 kernels of size k1 x 1]
          self.input_channels = input_channels
          self.conv = nn.Conv1d(1, n_kernels, kernel_size)
          self.pool = nn.MaxPool1d(pool_size)
          self.features_size = self._get_final_flattened_size()
          self.fc1 = nn.Linear(self.features_size, n4)
          self.fc2 = nn.Linear(n4, n_classes)
          self.apply(self.weight_init)

     def forward(self, x):
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)]
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.tanh(self.pool(x))
        x = x.view(-1, self.features_size)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

criterion = nn.CrossEntropyLoss()
