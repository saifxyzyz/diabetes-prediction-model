import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values
    scaler = StandardScaler()
    X= scaler.fit_transform(X)
    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X,y)
    data = np.hstack((X, np.reshape(y, (-1,1))))
    return data, X, y


def plot_histograms(dataframe, feature_list):
    for label in feature_list:
        plt.figure(figsize=(6,4))
        plt.hist(dataframe[dataframe['class'] == 1][label], color='red', label="Diabetic", alpha = 0.7, density=True)
        plt.hist(dataframe[dataframe['class'] == 0][label], color='blue', label="Not Diabetic", alpha = 0.7, density=True)
        plt.title(label)
        plt.ylabel('Probability')
        plt.xlabel(label)
        plt.legend()
        plt.show()

# For tensorflow

def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross entropy")
    plt.legend()
    plt.grid(True)
    plt.show()
