import torch
import torch.nn as nn

def MAE(y_true, y_pred):
  return torch.mean(torch.absolute(y_pred - y_true))

def coefficient_determination(y_true, y_pred):
  nom = torch.sum(torch.square(y_true - y_pred))
  denom = torch.sum(torch.square(y_true - torch.mean(y_true)))
  return 1 - nom / denom

class SE_Bottleneck(nn.Module):
  def __init__(self, in_shape, filters, ratio=16):
    super().__init__()
    self.filters = filters
    in_channels, in_width, in_height = in_shape

    self.avg2d = nn.AvgPool2d(kernel_size=(in_width, in_height))
    self.linear1 = nn.Linear(in_channels, filters//ratio)
    self.linear2 = nn.Linear(filters//ratio, filters)
  
  def forward(self, input_tensor):
    x = self.avg2d(input_tensor)
    x = nn.Flatten(start_dim=1)(x)
    x = self.linear1(x)
    nn.ReLU(inplace=True)(x)

    x = self.linear2(x)
    x = nn.Sigmoid()(x)

    x = torch.reshape(x, (-1, self.filters, 1, 1))

    return x * input_tensor

class SEResNet_Bottleneck(nn.Module):
  def __init__(self, in_shape, filters):
    super().__init__()
    in_channels, _, _ = in_shape
    
    self.conv2d1 = nn.Conv2d(in_channels, filters, kernel_size=(1, 1))
    self.bn1 = nn.BatchNorm2d(filters)

    self.se = SE_Bottleneck(in_shape, filters)

    self.conv2d2 = nn.Conv2d(in_channels, filters, kernel_size=(1, 1))
    self.bn2 = nn.BatchNorm2d(filters)
  
  def forward(self, input_tensor):
    x = self.conv2d1(input_tensor)
    x = self.bn1(x)
    nn.ReLU(inplace=True)(x)

    x = self.conv2d2(x)
    x = self.bn2(x)

    x = self.se(x)

    x = x + input_tensor

    return nn.ReLU()(x)

class GAiA_Network(nn.Module):
  def __init__(self, in_shape, filters, nb_blocks=4):
    super().__init__()

    in_channels, in_width, in_height = in_shape

    self.conv2d1 = nn.Conv2d(in_channels, filters, kernel_size=(1, 1))
    self.bn1 = nn.BatchNorm2d(filters)

    self.blocks = []
    in_shape = (filters, in_width, in_height)
    for _ in range(nb_blocks):
      self.blocks.append(SEResNet_Bottleneck(in_shape, filters))
    self.blocks = nn.Sequential(*self.blocks)
    
    self.conv2d2 = nn.Conv2d(filters, filters, kernel_size=(1, 1))
    self.bn2 = nn.BatchNorm2d(filters)

    self.output = nn.Linear(in_width*in_height*filters, 1)
  
  def forward(self, input_tensor):
    x = self.conv2d1(input_tensor)
    x = self.bn1(x)
    nn.ReLU(inplace=True)(x)

    x = self.blocks(x)

    x = self.conv2d2(x)
    x = self.bn2(x)
    nn.ReLU(inplace=True)(x)

    x = nn.Flatten(start_dim=1)(x)

    return self.output(x)