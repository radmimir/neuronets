import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import xlrd


def plot1(x, y, number_train):  # простой график двумерный
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('TIME')
    ax.set_ylabel('CONCENT')
    ax.set_title('Тр {:d}'.format(number_train))
    ax.grid()
    filename = 'results/{:d}.png'.format(number_train)
    plt.savefig(filename, dpi=96)
    plt.gca()
    plt.show()



def plot_results(history):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')

    plt.figure()
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('model mse')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')

    plt.figure()
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('model mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()
