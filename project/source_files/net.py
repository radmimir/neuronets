import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
import numpy as np
import xlrd
import math
from keras import backend as K
from mpl_toolkits.mplot3d import Axes3D
from keras.utils import plot_model
import pandas as pd

# импортируем бэкенд Agg из matplotlib для сохранения графиков на диск
import matplotlib

# matplotlib.use("Agg")

# подключаем необходимые пакеты
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.utils.generic_utils import get_custom_objects
from sklearn.metrics import classification_report
from keras import Sequential
from keras.layers import *
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

train_sheet = '#29'
test_sheet = '#30'


def read_xl(sheet, last):
    a = xlrd.open_workbook("dataset_last_changed.xlsx")
    sheet = a.sheet_by_name(sheet)
    first_ind = 0
    CO2 = sheet.col_values(0, first_ind, last)
    TTOPOIL = sheet.col_values(1, first_ind, last)
    TIME = sheet.col_values(2, first_ind, last)
    sheet.col_values(0)
    n = len(CO2)
    m = len(TTOPOIL)
    k = len(TIME)
    try:
        for i in range(n):
            if CO2[i] == '' or TIME[i] == '' or TTOPOIL[i] == '':
                CO2.pop(i)
                TIME.pop(i)
                TTOPOIL.pop(i)
                if i == len(CO2) - 1:
                    break
    except IndexError:
        print(i, len(CO2), len(TIME), len(TTOPOIL))

    return TIME, CO2, TTOPOIL


def net():
    # training dataset
    TIME, CO2, TTOPOIL = read_xl(train_sheet, 4815)
    first = float(TIME[0])
    n = len(TIME)
    TIME = np.array([x - first for x in TIME])
    x_train = np.array([TTOPOIL, TIME], dtype='float')
    x_train = x_train.reshape(4815, 2)
    y_train = np.array(CO2, dtype='float')
    y_train = y_train.reshape(4815, 1)
    # testing dataset
    """TIME, CO2, TTOPOIL = read_xl(test_sheet, 4815)
    first = float(TIME[0])
    TIME = np.array([x - first for x in TIME])
    x_test = np.array([TTOPOIL, TIME], dtype='float')
    x_test = x_test.reshape(4815, 2)
    y_test = np.array(CO2, dtype='float')
    y_test = y_test.reshape(4815, 1)"""
    (x_train1, x_test) = train_test_split(x_train, test_size=0.25, random_state=42)
    (y_train1, y_test) = train_test_split(y_train, test_size=0.25, random_state=42)
    model = create_mlp(2, x_train, regress=True)
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse', 'mae'])
    history = model.fit(x_train1, y_train1, validation_data=(x_test, y_test), epochs=30)
    predictions = model.predict(x_train)
    predictions = predictions.reshape(1, len(predictions))
    predictions = list(predictions[0])
    appr = linalg(TTOPOIL, TIME)
    x, y, z1, z2, z3 = [], [], [], [], []
    step = 20
    for i in range(4 * step, len(TIME), step):
        x.append(TIME[i])
        y.append(TTOPOIL[i])
        z1.append(mean([predictions[i], predictions[i - step], predictions[i - 2 * step], predictions[i - 3 * step],
                        predictions[i - 4 * step]]))
        z3.append(mean([appr[i], appr[i - step], appr[i - 2 * step], appr[i - 3 * step], appr[i - 4 * step]]))
    graph3d(TIME, TTOPOIL, z1, CO2, z3, x, y)
    plot_model(model, "model.png", True, True, expand_nested=True)
    # plot_results(history)


def mean(x):
    return sum(x) / len(x)


def activation1(x):
    # time = x[1]
    # print(y, x)
    return K.exp(x)
    # K.dot(K.elu(x), y)
    # return K.switch(K.less(x, 250), K.exp(x), K.zeros(x.shape()))


def activation2(x):
    # time = x[1]
    return K.switch(K.greater(x, 250), K.exp(-x), K.zeros(x.shape))


def activation3(x):
    # time = x[1]
    return K.switch(K.greater(x, 450), K.exp(x), K.zeros(x.shape))


def activation4(x):
    # time = x[1]
    return K.switch(K.greater(x, 650), K.exp(-x), K.zeros(x.shape))


def plot1(x, y1):  # простой график двумерный
    fig, ax = plt.subplots()
    ax.plot(x, y1)
    ax.set_xlabel('TIME')
    ax.set_ylabel('CONCENT')
    ax.set_title('Тр 35')
    ax.grid()
    filename = 'results/35.png'
    plt.savefig(filename, dpi=96)
    plt.gca()
    # plt.show()


def create_mlp(dim, x_train, regress=False):
    # define our MLP networkmodel = Sequential()
    model = Sequential()
    model.add(Dense(64, input_dim=2))
    # for i in range(50):
    # .add(Dense(64))
    # model.add(Activation('tanh'))  # , activation='sigmoid'))
    """get_custom_objects().update({'activation1': Activation(activation1)})
    get_custom_objects().update({'activation2': Activation(activation2)})
    get_custom_objects().update({'activation3': Activation(activation3)})
    get_custom_objects().update({'activation4': Activation(activation4)})"""
    """model.add(Dense(64, input_dim=2, activation=K.elu, activity_regularizer=keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(64, activation="elu", activity_regularizer=keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(64, activation="elu", activity_regularizer=keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(64, activation="elu", activity_regularizer=keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())"""
    """
    model.add(Dense(16, activation="selu", activity_regularizer=keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(64, input_dim=2, activation="selu", activity_regularizer=keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(16, activation="selu", activity_regularizer=keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())"""
    model.add(Dense(64, activation=K.elu))
    model.add(BatchNormalization())
    model.add(Dense(64, activation=K.elu))
    model.add(BatchNormalization())
    model.add(Dense(64, activation=K.elu))
    model.add(BatchNormalization())
    model.add(Dense(64, activation=K.elu))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.add(Activation('linear'))

    """model = Sequential()
    model.add(Dense(8, input_dim=2, activation="relu"))
    
    model.add(Dense(4, activation="relu"))

    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))"""

    """model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(1, n)),
        tf.keras.layers.Dense(128, activation=tf.nn.sigmoid),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.sigmoid)
    ])"""
    # return our model
    return model


def graph3d(x1, x2, yy, q, appr, x1new, x2new):
    fig = plt.figure()
    ax = Axes3D(fig)
    # TIME, CO2, TTOPOIL = read_xl()
    # ax.plot3D(x1new, x2new, yy,
    #          color='red', linewidth=3.0)  # CO2, TTOPOIL, TIME) построение графика регрессии из предсказанных значений
    ax.plot3D(x1, x2, q, linewidth=0.5)  # построение графика исходных
    ax.plot3D(x1new, x2new, appr, color='green', linewidth=3.0)  # CO2, TTOPOIL, TIME) построение графика аппроксимации
    ax.legend(['Прогноз нейросети', 'Исх. данные', 'Анал. модель'])
    xs = np.zeros(1000)
    ys = np.zeros(1000)
    zs = np.array([i for i in range(42700, 43700)])
    # ax.plot3D(xs, ys, zs)
    ax.set_xlabel('TOIL')
    ax.set_ylabel('TIME')
    ax.set_zlabel('CONCENT')
    ax.set_label('29')
    ax.view_init(0, -90)
    ax.view_init(0, 180)
    filename = 'results/step_180.png'
    plt.savefig(filename, dpi=96)
    ax.view_init(0, -90)
    filename = 'results/step_-90.png'
    plt.savefig(filename, dpi=96)

    # Мы собираемся сделать 20 графиков, для 20 разных углов
    """for file in os.listdir('frames'):
        file_path = os.path.join('frames', file)
        os.unlink(file_path)
    for angle in range(70, 270, 2):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot3D(x1new, x2new, yy,
                  color='red',
                  linewidth=3.0)  # CO2, TTOPOIL, TIME) построение графика регрессии из предсказанных значений
        ax.plot3D(x1, x2, q, linewidth=0.5)  # построение графика исходных
        ax.plot3D(x1new, x2new, appr, color='green',
                  linewidth=3.0)  # CO2, TTOPOIL, TIME) построение графика аппроксимации
        ax.set_xlabel('TOIL')
        ax.set_ylabel('TIME')
        ax.set_zlabel('CONCENT')
        ax.legend(['Прогноз нейросети', 'Исх. данные', 'Анал. модель'])

        ax.view_init(30, angle)

        filename = 'frames/step' + str(angle) + '.png'
        plt.savefig(filename, dpi=96)
        plt.gca()"""
    # plt.show()


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


def mul_regr(TTOPOIL, TIME):  # решение Матричного уравнения AX = Y, X = QR, Мультипликативная регрессия
    a = xlrd.open_workbook("dataset_last_changed.xlsx")
    sheet = a.sheet_by_index(4)
    X, Y, B = [], [], []
    x1, x2 = [], []
    q = []
    yy = []
    first = sheet.cell(0, 2).value - 1
    for i in range(4815):
        b = sheet.row_values(i)
        if '' in b or None in b:
            continue
        b[1] += 273  # температура
        b[2] -= first  # время(относительное)
        x1.append(b[1])
        x2.append(b[2])
        q.append(b[0])
        b = np.log(b)
        B.append(b[0])
        b[0] = 1
        X.append(list(b))
    Q, R = np.linalg.qr(X)
    Y = np.dot(Q.T, B)
    res = np.linalg.solve(R, Y)
    res[0] = math.exp(res[0])  # коэффициенты a, температура масла в кельвинах, время относительное
    return res, x1, x2, q


def linalg(TTOPOIL, TIME):
    a, x1, x2, q = mul_regr(TTOPOIL, TIME)  # exp_regr()
    n = len(x1)
    yy = []
    for i in range(n):
        yy.append(a[0] * (x1[i] ** a[1]) * (x2[i] ** a[2]))
        # yy.append(a[0] * (math.e ** (a[1] * x1[i] + a[2] * x2[i]))) exp
    return yy


net()
