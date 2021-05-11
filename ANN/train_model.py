import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(physical_devices)
except:
    print('No Cuda GPU detected.')


mode = 'batch'
offset = 0
max_idx = 55
engine = 'Stockfish 13'
model_path = '../Models/SF_model_batch_65M'


def buildAndCompile(shape):
    input = tf.keras.Input(shape=(shape,))
    output = layers.Dense(shape, activation='relu')(input)
    output = layers.Dense(64, activation='relu')(output)
    output = layers.Dense(1)(output)
    model = keras.Model(inputs=input, outputs=output)
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model


def save_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 300])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('history.png')


def saveTFModel(model, output_path):
    s = ""
    for layer in model.layers:
        w = layer.get_weights()
        if len(w) > 1:
            s += f"layer {layer.get_config()['activation']}\n"
            weights = w[0]
            bias = w[1]

            dimensions = len(weights)
            nb_neurons = len(weights[0])

            if nb_neurons == 1:
                for i in range(dimensions):
                    s += str(weights[i][0]) + " "
                s += str(bias[0]) + "\n"

            else:
                for i in range(nb_neurons):
                    w = []
                    for j in range(dimensions):
                        s += str(weights[j][i]) + " "
                    s += str(bias[i]) + "\n"
    f = open(output_path, "w")
    f.write(s)
    f.close()


rebuild = True
model = None

if rebuild:
    model = buildAndCompile(131)
else:
    model = keras.models.load_model('../Models/SF_model_batch_55M')

print(model.summary())


if mode == 'all':

    X = []
    y = []

    for i in range(max_idx):
        dataframe_encoded = pd.read_csv(
            'Datasets/' + engine + '/dataset' + str(i+1) + '.csv')
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
    save_loss(history)
    model.save(model_path, save_format='tf')

else:

    for i in range(offset, max_idx):
        dataframe_encoded = pd.read_csv(
            'Datasets/' + engine + '/dataset' + str(i+1) + '.csv')
        features = dataframe_encoded.columns[:-1]
        cps = dataframe_encoded.columns[-1]
        X = dataframe_encoded[features].values
        y = dataframe_encoded[cps].values

        history = model.fit(X, y, validation_split=0.1, verbose=0, epochs=50)
        model.save(model_path, save_format='tf')
        print(f'Training finished on dataset: dataset{i+1}.csv')

    save_loss(history)
    print("Model saved in: ", model_path)

saveTFModel(model, model_path + ".nn")
