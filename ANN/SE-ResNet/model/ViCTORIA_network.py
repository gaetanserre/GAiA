import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
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
    x1 = self.conv2a(input_tensor)
    x = self.bna(x1)

    x = self.conv2b(x)
    x = self.bnb(x)

    x = self.se(x)

    x = layers.Add()([x, x1])
    return tf.nn.relu(x)

class ViCTORIA_Network(tf.keras.Model):
  def __init__(self, filters, nb_blocks=6):
    super(ViCTORIA_Network, self).__init__(name="ViCTORIA")
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
    x = tf.nn.relu(x)

    for block in self.blocks:
      x = block(x)

    x = self.conv2b(x)
    x = self.bnb(x)
    x = tf.nn.relu(x)

    x = layers.Flatten()(x)

    return self.dense(x)