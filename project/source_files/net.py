import numpy as np
from keras import backend as K
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

# подключаем необходимые пакеты
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import *
from keras.layers.core import Dense

from functions import tools, regression, graphs

labels = ['TOIL', 'TIME', 'CONCENT',
          'Прогноз нейросети', 'Исх. данные', 'Анал. модель']

train_sheet = '#29'
test_sheet = '#30'
number_train = int(train_sheet[1:])


def net():
    # training dataset
    TIME, CO2, TTOPOIL, n = tools.read_xl(train_sheet)
    n = len(TIME)
    x, y, z1, z2, z3 = [], [], [], [], []
    x_train = np.array([TTOPOIL, TIME], dtype='float')
    x_train = x_train.transpose()
    y_train = np.array(CO2, dtype='float')
    y_train = y_train.transpose()
    # testing dataset
    """TIME, CO2, TTOPOIL = read_xl(test_sheet, 4815)
    first = float(TIME[0])
    TIME = np.array([x - first for x in TIME])
    x_test = np.array([TTOPOIL, TIME], dtype='float')
    x_test = x_test.reshape(4815, 2)
    y_test = np.array(CO2, dtype='float')
    y_test = y_test.reshape(4815, 1)"""
    (x_train1, x_train2, y_train1, y_train2) = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    poly = PolynomialFeatures(degree=5)
    X_1 = poly.fit_transform(x_train)
    clf = linear_model.LinearRegression()
    history = clf.fit(X_1, y_train)
    x_train1 = poly.fit_transform(x_train)
    predictions1 = clf.predict(x_train1)
    predictions1 = predictions1.reshape(1, len(predictions1))
    predictions1 = list(predictions1[0])
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
    z1 = predictions1
    z3 = regression.linalg(TTOPOIL, TIME, CO2)
    print("z1=", len(TTOPOIL))
    print("z3=", len(z3))
    graphs.graph3d(TIME, TTOPOIL, z1, CO2, z3, x, y, labels, number_train)
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
