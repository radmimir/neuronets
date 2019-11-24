import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import xlrd


def read_xl():
    a = xlrd.open_workbook("dataset_last (copy).xlsx")
    sheet = a.sheet_by_index(4)
    CO2 = sheet.col_values(5, 172, 5000)
    TIME = sheet.col_values(0, 172, 5000)
    TTOPOIL = sheet.col_values(17, 172, 5000)
    n = len(CO2)
    m = len(TIME)
    k = len(TTOPOIL)
    try:
        for i in range(n):
            if CO2[i] == '' or TIME[i] == '' or TTOPOIL[i] == '':
                CO2.pop(i)
                TIME.pop(i)
                TTOPOIL.pop(i)
    except IndexError:
        print(i, CO2[i - 1])
    return TIME, CO2, TTOPOIL
    # for d in mydict:
    #    mydict[d] = []
    # for rownum in range(1, sheet.nrows):
    #    for l in labels:
    #        dick[l].append(sheet.row_values(rownum))


def mul_regr():  # решение Матричного уравнения AX = Y, X = QR, Мультипликативная регрессия
    a = xlrd.open_workbook("dataset_last_changed.xlsx")
    sheet = a.sheet_by_index(4)
    X, Y, B = [], [], []
    x1, x2 = [], []
    q = []
    yy = []
    first = sheet.cell(0, 2).value - 1
    for i in range(5000):
        b = sheet.row_values(i)
        if '' in b or None in b:
            continue
        b[1] += 273 # температура
        b[2] -= first # время(относительное)
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

    # res = np.linalg.solve(X, Y)
    # return res


def graph3d(x1, x2, yy, q):
    fig = plt.figure()
    ax = Axes3D(fig)
    # TIME, CO2, TTOPOIL = read_xl()
    ax.plot3D(x1, x2, yy,color='red')  # CO2, TTOPOIL, TIME) построение графика аппроксимации
    ax.scatter(x1, x2, q)  # построение графика исходных
    xs = np.zeros(1000)
    ys = np.zeros(1000)
    zs = np.array([i for i in range(42700, 43700)])
    # ax.plot3D(xs, ys, zs)
    ax.set_xlabel('TOIL')
    ax.set_ylabel('TIME')
    ax.set_zlabel('CONCENT')
    plt.show()


def exp_regr():  # решение Матричного уравнения AX = Y, X = QR, Экспоненциальная регрессия
    f = xlrd.open_workbook("dataset_last_changed.xlsx")
    sheet = f.sheet_by_index(4)
    X, Y, B = [], [], []
    x1, x2 = [], []
    q = []
    yy = []
    first = sheet.cell(0, 2).value - 1
    for i in range(5000):
        b = sheet.row_values(i)
        if '' in b or None in b:
            continue
        b[1] += 273
        b[2] -= first
        x1.append(b[1])
        x2.append(b[2])
        q.append(b[0])
        # b = np.log(b) для экспоненты
        B.append(np.log(b[0]))
        b[0] = 1
        X.append(list(b))
    Q, R = np.linalg.qr(X)
    Y = np.dot(Q.T, B)
    a = np.linalg.solve(R, Y)
    a[0] = math.exp(a[0])  # коэффициенты a, температура масла в кельвинах, время относительное
    return a, x1, x2, q


def linalg():
    a, x1, x2, q = mul_regr() # exp_regr()
    n = len(x1)
    yy = []
    for i in range(n):
        yy.append(a[0] * (x1[i] ** a[1]) * (x2[i] ** a[2]))
        # yy.append(a[0] * (math.e ** (a[1] * x1[i] + a[2] * x2[i]))) exp
    graph3d(x1, x2, yy, q)
    # print(CO2)

linalg()