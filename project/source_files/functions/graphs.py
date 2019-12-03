# module for plotting 3 - dimensional plots from numpy arrays
import os
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from . import tools


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


def create_gif(x, y, z, labels):  # 3 графика 3*tuple(
    n = len(x)
    x1, x2, x3 = x
    y1, y2, y3 = y
    z1, z2, z3 = z
    # Мы собираемся сделать 20 графиков, для 20 разных углов
    for file in os.listdir('frames'):
        file_path = os.path.join('frames', file)
        os.unlink(file_path)
    for angle in range(70, 270, 2):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot3D(x1, y1, z1,
                  color='red',
                  linewidth=3.0)  # CO2, TTOPOIL, TIME) построение графика регрессии из предсказанных значений
        ax.plot3D(x2, y2, z2, linewidth=0.5)  # построение графика исходных данных
        ax.plot3D(x3, y3, z3, color='green',
                  linewidth=3.0)  # CO2, TTOPOIL, TIME) построение графика аппроксимации
        ax.set_xlabel(labels[0])
        ax.set_xlabel(labels[1])
        ax.set_xlabel(labels[2])
        ax.legend(labels[3:6])

        ax.view_init(30, angle)

        filename = 'frames/step' + str(angle) + '.png'
        plt.savefig(filename, dpi=96)
        plt.gca()


def graph3d(x1, x2, yy, q, appr, x1new, x2new, labels, number_train):
    fig = plt.figure()
    ax = Axes3D(fig)
    q = tools.gauss(q)
    yy = tools.gauss(yy)
    appr = tools.gauss(appr)
    x2 = ndimage.gaussian_filter(x2, sigma=100., order=0)
    ax.plot3D(x1, x2, yy,
              color='red', linewidth=1.0)  # CO2, TTOPOIL, TIME) # построение графика исходных
    ax.plot3D(x1, x2, appr, color='green', linewidth=2.0)  # CO2, TTOPOIL, TIME) построение графика аппроксимации
    ax.scatter(x1[::30], x2[::30], q[::30], s=20, c='black')
    ax.legend(labels[3:6])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_label(str(number_train))
    ax.view_init(0, -90)
    ax.view_init(0, 180)
    filename = 'results/step_180.png'
    plt.savefig(filename, dpi=96)
    ax.view_init(0, -90)
    filename = 'results/step_-90.png'
    plt.savefig(filename, dpi=96)
    ax.view_init(0, -90)
    plt.show()