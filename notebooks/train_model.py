import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('No Cuda GPU detected.')
    
    
    
mode = 'batch'
offset = 0
max_idx = 32
engine = 'Stockfish 13'
model_path = '../Models/SF_model_batch_32M'
    
    
    
    
def plotPred (y_true, preds):
    ymin = np.min(y_true)
    ymax = np.max(y_true)
    plt.scatter(y_true, preds, label=f'score: {r2_score(y_true, preds)}')
    plt.plot([ymin, ymax], [ymin, ymax], '-.', color='red', label="predicted values = true values")
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.legend()
    
    
def buildAndCompile(shape):
    input = tf.keras.Input(shape=(shape,))
    output = layers.Dense(shape, activation='relu')(input)
    output = layers.Dense(64, activation='relu')(output)
    output = layers.Dense(1)(output)
    model = keras.Model(inputs=input, outputs=output)
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 300])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


rebuild = True
model = None

if rebuild:
    model = buildAndCompile(131)
else:
    model = keras.models.load_model('../Models/SF_model_batch_32M')
    
print(model.summary())


if mode == 'all':

    X = []
    y = []

    for i in range(max_idx):
        dataframe_encoded = pd.read_csv('Datasets/' + engine + '/dataset' + str(i+1) + '.csv')
        features = dataframe_encoded.columns[:-1]
        cps = dataframe_encoded.columns[-1]

        if len(X) == 0:
            X = dataframe_encoded[features].values
            y = dataframe_encoded[cps].values
        else:
            X = np.append(X, dataframe_encoded[features].values, axis=0)
            y = np.append(y, dataframe_encoded[cps].values)

    print(X.shape, y.shape)

    history = model.fit(X, y, validation_split=0.1, verbose=0, epochs=50)
    plot_loss(history)
    model.save(model_path, save_format='tf')
    
else:
    
    for i in range(offset, max_idx):
        dataframe_encoded = pd.read_csv('Datasets/' + engine + '/dataset' + str(i+1) + '.csv')
        features = dataframe_encoded.columns[:-1]
        cps = dataframe_encoded.columns[-1]
        X = dataframe_encoded[features].values
        y = dataframe_encoded[cps].values

        history = model.fit(X, y, validation_split=0.1, verbose=0, epochs=50)
        print(f'Training finished on dataset: dataset{i+1}.csv')
        model.save(model_path, save_format='tf')
    
    plot_loss(history)
    