import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import random
import librosa, IPython
import librosa.display as lplt
import tensorflow as tf
import keras as k
from sklearn.preprocessing import StandardScaler


from joblib import Parallel, delayed
import joblib
seed = 12
np.random.seed(seed)

class myCallback(k.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy') > ACCURACY_THRESHOLD):
            print("\n\nStopping training as we have reached %2.2f%% accuracy!" %(ACCURACY_THRESHOLD*100))
            self.model.stop_training = True

def trainModel(model, epochs, optimizer):
    batch_size = 128
    callback = myCallback()
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
    )
    return model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=epochs,
                     batch_size=batch_size, callbacks=[callback])
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
    df = pd.read_csv('src/Data/features_3_sec.csv')
    # map labels to index
    label_index = dict()
    index_label = dict()
    for i, x in enumerate(df.label.unique()):
        label_index[x] = i
        index_label[i] = x
    print(label_index)
    print(index_label)

    df.label = [label_index[l] for l in df.label]
    # shuffle samples
    df_shuffle = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # remove irrelevant columns
    df_shuffle.drop(['filename', 'length'], axis=1, inplace=True)
    df_y = df_shuffle.pop('label')
    df_X = df_shuffle

    # split into train dev and test
    X_train, df_test_valid_X, y_train, df_test_valid_y = skms.train_test_split(df_X, df_y, train_size=0.7,
                                                                               random_state=seed, stratify=df_y)
    X_dev, X_test, y_dev, y_test = skms.train_test_split(df_test_valid_X, df_test_valid_y, train_size=0.66,
                                                         random_state=seed, stratify=df_test_valid_y)

    scaler = skp.StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_dev = pd.DataFrame(scaler.transform(X_dev), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

    scaler_filename = "scaler.save"
    joblib.dump(scaler, scaler_filename)

    ACCURACY_THRESHOLD = 0.94

    model_1 = k.models.Sequential([
        k.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        k.layers.Dropout(0.2),

        k.layers.Dense(128, activation='relu'),
        k.layers.Dropout(0.2),

        k.layers.Dense(64, activation='relu'),
        k.layers.Dropout(0.2),

        k.layers.Dense(10, activation='softmax'),
    ])
    print(model_1.summary())
    model_1_history = trainModel(model=model_1, epochs=70, optimizer='adam')
    #plot_history(history=model_1_history)
    # Save the Model using pickle
    joblib.dump(model_1, 'final.pkl')

    # Load the model from the file
    Model_from_joblib = joblib.load('final.pkl')

    # Use the loaded model to make predictions
    Model_from_joblib.predict(X_test)
