import numpy as np
from keras import backend as K
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from scipy import ndimage

# подключаем необходимые пакеты
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import *
from keras.layers.core import Dense

from functions import tools, regression, graphs

labels = ['TIME', 'TOIL', 'CONCENT CO2',
          'Нейросетевая модель', 'Анал. модель', 'Исх. данные']
train_sheet = '#29'
test = '#30', '#31'
test_sheet = '#30'
trans_3_labels = ['TIME', 'TOIL', 'CONCENT CO2',
                  'Тр 29', 'Тр 30', 'Тр 31']
number_train = int(train_sheet[1:])


def net():
    # training dataset
    TIME, CO2, TTOPOIL, n = tools.read_xl(train_sheet)
    TIME_30, CO2_30, TTOPOIL_30, n_30 = tools.read_xl(test[0])
    TIME_31, CO2_31, TTOPOIL_31, n_31 = tools.read_xl(test[1])
    n = len(TIME)
    x, y, z1, z2, z3 = [], [], [], [], []
    x_train = np.array([TTOPOIL, TIME], dtype='float').transpose()
    y_train = np.array(CO2, dtype='float').transpose()

    # testing 30
    x_test_30 = np.array([TTOPOIL_30, TIME_30], dtype='float').transpose()
    y_test_30 = np.array(CO2_30, dtype='float').transpose()

    # testing 31
    x_test_31 = np.array([TTOPOIL_31, TIME_31], dtype='float').transpose()
    y_test_31 = np.array(CO2_31, dtype='float').transpose()

    # (x_train1, x_train2, y_train1, y_train2) = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    poly = PolynomialFeatures(degree=6)
    X_1 = poly.fit_transform(x_train)
    clf = linear_model.LinearRegression(normalize=True)
    history = clf.fit(X_1, y_train)
    metric1 = clf.score(X_1, y_train)
    print(metric1)
    # predictions
    predictions_29 = clf.predict(X_1)  # по обучающей выборке
    predictions_29 = predictions_29.reshape(1, len(predictions_29))
    predictions_29 = list(predictions_29[0])
    # 30
    x_test_30_ = poly.fit_transform(x_test_30)
    predictions_30 = clf.predict(x_test_30_)
    predictions_30 = predictions_30.reshape(1, len(predictions_30))
    predictions_30 = list(predictions_30[0])
    metric_30 = clf.score(x_test_30_, y_test_30)
    print(metric_30)
    # 30
    x_test_31_ = poly.fit_transform(x_test_31)
    predictions_31 = clf.predict(x_test_31_)
    predictions_31 = predictions_31.reshape(1, len(predictions_31))
    predictions_31 = list(predictions_31[0])
    metric_31 = clf.score(x_test_31_, y_test_31)
    print(metric_31)
    # старая модель по keras
    """model = create_mlp(2, x_train, regress=True)
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse', 'mae'])
    history = model.fit(x_train1, y_train1, validation_data=(x_train2, y_train2), epochs=30)
    predictions = model.predict(x_train)
    predictions = predictions.reshape(1, len(predictions))
    predictions = list(predictions[0])"""
    step = 20
    # код - попытка сглаживающим окном решить проблему разброса точек
    """for i in range(1, n):
        if abs(z1[i] - z1[i - 1]) > 100:
            z1[i] = z1[i - 1]
    print(z1[4000:4615])"""
    """for i in range(4 * step, len(TIME), step):
        x.append(TIME[i])
        y.append(TTOPOIL[i])
        z1.append(mean([predictions[i], predictions[i - step], predictions[i - 2 * step], predictions[i - 3 * step],
                        predictions[i - 4 * step]]))
        z3.append(mean([appr[i], appr[i - step], appr[i - 2 * step], appr[i - 3 * step], appr[i - 4 * step]]))"""
    z1 = predictions_29
    z3 = regression.linalg(TTOPOIL, TIME, CO2)
    graphs.graph3d(TIME, TTOPOIL, z1, CO2, z3, x, y, labels, number_train)
    x = list(map(tools.gauss, [TIME, TIME_30, TIME_31]))
    for i in range(len(y)):
        y[i] = ndimage.gaussian_filter(y[i], sigma=100., order=0)
    z = list(map(tools.gauss, [CO2, CO2_30, CO2_31]))
    # построение по 29 гифки
    x = list(map(tools.gauss, [TIME, TIME, TIME]))
    y = list(map(tools.gauss, [TTOPOIL, TTOPOIL, TTOPOIL]))
    for i in range(len(y)):
        y[i] = ndimage.gaussian_filter(y[i], sigma=100., order=0)
    z = [z1, z3]
    z = list(map(tools.gauss, z))
    z.insert(1, CO2)

    graphs.create_gif(x, y, z, labels)
    # graphs.graph3d_3trans(x, y, z, labels=trans_3_labels)

    # plot_model(model, "model.png", True, True, expand_nested=True)
    # plot_results(history)


def mean(x):
    return sum(x) / len(x)


def activation1(x):
    return K.exp(x)


def activation2(x):
    return K.switch(K.greater(x, 250), K.exp(-x), K.zeros(x.shape))


def activation3(x):
    return K.switch(K.greater(x, 450), K.exp(x), K.zeros(x.shape))


def activation4(x):
    return K.switch(K.greater(x, 650), K.exp(-x), K.zeros(x.shape))


def create_mlp(dim, x_train, regress=False):  # simple MLP for regression
    model = Sequential()
    model.add(Dense(64, input_dim=2))
    model.add(Dense(64, activation=K.elu))
    model.add(BatchNormalization())
    model.add(Dense(64, activation=K.elu))
    model.add(BatchNormalization())
    model.add(Dense(64, activation=K.elu))
    model.add(BatchNormalization())
    model.add(Dense(64, activation=K.elu))
    model.add(BatchNormalization())
    if regress:
        model.add(Dense(1))
        model.add(Activation('linear'))
    else:
        model.add(Activation('sigmoid'))
    return model


net()
