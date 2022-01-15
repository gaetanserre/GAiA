import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def coeff_determination(y_true, y_pred):
  SS_res =  K.sum(K.square( y_true-y_pred ))
  SS_tot = K.sum(K.square( y_true - K.mean(y_true)))
  return ( 1 - SS_res/(SS_tot + K.epsilon()) )

class SEBlock(layers.Layer):
  def __init__(self, shape, ratio=16):
    super(SEBlock, self).__init__()
    self.dense1 = layers.Dense(shape//ratio, activation="relu")
    self.dense2 = layers.Dense(shape, activation="sigmoid")
  
  def call(self, input_tensor):
    x = layers.GlobalAveragePooling2D()(input_tensor)
    x = self.dense1(x)
    x = self.dense2(x)
    return layers.Multiply()([input_tensor, x])


class SE_ResNetBlock(layers.Layer):
  def __init__(self, filters):
    super(SE_ResNetBlock, self).__init__()
    self.conv2a = layers.Conv2D(filters, (1,1))
    self.bna = layers.BatchNormalization()

    self.se = SEBlock(filters)

    self.conv2b = layers.Conv2D(filters, (1,1))
    self.bnb = layers.BatchNormalization()

  def call(self, input_tensor):
    x = self.conv2a(input_tensor)
    x = self.bna(x)
    x = layers.ReLU()(x)

    x = self.conv2b(x)
    x = self.bnb(x)

    x = self.se(x)

    x = layers.Add()([x, input_tensor])
    return layers.ReLU()(x)

class GAiA_Network(tf.keras.Model):
  def __init__(self, filters, nb_blocks=6):
    super(GAiA_Network, self).__init__(name="GAiA")
    self.filters = filters
    self.nb_blocks = nb_blocks

    self.conv2a = layers.Conv2D(self.filters, (1, 1))
    self.bna = layers.BatchNormalization()

    self.blocks = []
    for _ in range(self.nb_blocks):
      self.blocks.append(SE_ResNetBlock(self.filters))

    self.conv2b = layers.Conv2D(self.filters, (1, 1))
    self.bnb = layers.BatchNormalization()
    
    self.dense = layers.Dense(1, activation="linear")
  
  def call(self, input_tensor):
    x = self.conv2a(input_tensor)
    x = self.bna(x)
    x = layers.ReLU()(x)

    for block in self.blocks:
      x = block(x)

    x = self.conv2b(x)
    x = self.bnb(x)
    x = layers.ReLU()(x)

    x = layers.Flatten()(x)

    return self.dense(x)

def GAiA_Network(shape, hyperparameters):
  filters = hyperparameters["filters"]
  inputs = tf.keras.Input(shape, name="input")
  x = layers.Conv2D(filters, (1, 1))(inputs)
  x = layers.BatchNormalization()(x)
  #x = layers.ReLU()(x)

  for _ in range(hyperparameters["nb_blocks"]):
    x_old = x
    x = layers.Conv2D(filters, (1,1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (1,1))(x)
    x = layers.BatchNormalization()(x)

    # SE
    x_se = layers.GlobalAveragePooling2D()(x)
    x_se = layers.Dense(filters//16, activation="relu")(x_se)
    x_se = layers.Dense(filters, activation="sigmoid")(x_se)
    x = layers.Multiply()([x, x_se])

    x = layers.Add()([x, x_old])
    x = layers.ReLU()(x)

  x = layers.Conv2D(filters, (1, 1))(x)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)

  x = layers.Flatten()(x)

  x = layers.Dense(1, activation="linear", name="output")(x)


  model = tf.keras.Model(inputs=inputs, outputs=x)
  model._name = "GAiA"
  return model

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