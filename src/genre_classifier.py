import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers

DATASET_PATH = "data.json"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
            data = json.load(fp)

    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    return inputs, targets

def plot_history(history):
    fig, axis = plt.subplots(2)

    #create accuracy subplot
    axis[0].plot(history.history["accuracy"], label="train accuracy")
    axis[0].plot(history.history["val_accuracy"], label="test accuracy")
    axis[0].set_ylabel("Accuracy")
    axis[0].legend(loc="lower right")
    axis[0].set_title("Accuracy eval")

    #create error subplot
    axis[1].plot(history.history["loss"], label="train error")
    axis[1].plot(history.history["val_loss"], label="test error")
    axis[1].set_ylabel("Error")
    axis[1].set_xlabel("Epoch")
    axis[1].legend(loc="upper right")
    axis[1].set_title("Error eval")

    plt.show()

if __name__ == "__main__":
    inputs, targets = load_data(DATASET_PATH)

    #split the data
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3)

    #building the network architecture
    model = keras.Sequential([
        #input layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        #1st hidden layer
        keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.2),

        #2nd hidden layer
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.2),

        #3rd hidden layer
        keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.2),

        #4th hidden layer
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.2),

        #output layer
        keras.layers.Dense(10, activation="softmax")

    ])
    '''model = Sequential()
    model.add(Dense(units=256, activation='relu', input_dim=inputs_train.shape[1]))
    model.add(Dropout(0.2))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=len(np.unique(targets_train)), activation='softmax'))  # Adjusted to use np.unique for flexibility
    model.summary()'''

    #compile network
    optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    #train network
    history = model.fit(inputs_train, targets_train, validation_data=(inputs_test, targets_test), epochs=100, batch_size=128)

    plot_history(history=history)

