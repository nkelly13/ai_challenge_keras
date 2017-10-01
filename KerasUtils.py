import tensorflow as tf
import pickle as pkl
import numpy as np
import string

import keras
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def load_data(path, classes=10, rows=28, cols=28):
    if K.image_data_format() == 'channels_first':
        x_data = np.zeros([0,1,rows,cols])
    else:
        x_data = np.zeros([0,rows,cols,1])
        input_shape = (rows, cols, 1)

    y_data = np.zeros([0,classes])

    for index, a_char in enumerate(string.ascii_uppercase[0:10]):
        char_arrays = []
        with open("{}/{}.pickle".format(path,a_char),'rb') as f:
            char_data = pkl.load(f)
            f.close()
        char_data = char_data.reshape((char_data.shape[0],*input_shape))
        x_data = np.append(x_data,char_data,axis=0)
        char_labels = index*np.ones(char_data.shape[0])
        categorical_labels = keras.utils.to_categorical(char_labels,classes)
        y_data = np.append(y_data,categorical_labels,axis=0)
    x_data = x_data.astype('float32')
    x_data /= 255
    print('loaded data shape:', x_data.shape)
    return x_data, y_data, input_shape

def plot_confusion_matrix(cm, classes, plt, title='Confusion matrix'):
    plt.imshow(np.log(cm), interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    num_classes = len(classes)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
   