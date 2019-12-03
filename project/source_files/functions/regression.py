import numpy as np
import math


def exp_regr(TTOPOIL, TIME, conc):  # решение Матричного уравнения AX = Y, X = QR, Экспоненциальная регрессия
    n = len(TTOPOIL)
    X, Y, B = [], [], []
    x1, x2 = [], []
    q = []
    for i in range(n):
        b = [conc[i], TTOPOIL[i], TIME[i]]
        x1.append(b[1])
        x2.append(b[2])
        q.append(b[0])
        B.append(np.log(b[0]))
        b[0] = 1
        X.append(list(b))
    Q, R = np.linalg.qr(X)
    Y = np.dot(Q.T, B)
    a = np.linalg.solve(R, Y)
    a[0] = math.exp(a[0])  # коэффициенты a, температура масла в кельвинах, время относительное
    return a, x1, x2, q


def mul_regr(TTOPOIL, TIME, conc):  # решение Матричного уравнения AX = Y, X = QR, Мультипликативная регрессия
    n = len(TTOPOIL)
    X, Y, B = [], [], []
    x1, x2 = [], []
    q = []
    for i in range(n):
        b = [conc[i], TTOPOIL[i], TIME[i]]
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


def linalg(TTOPOIL, TIME, conc, exp=False):
    if exp:
        a, x1, x2, q = exp_regr(TTOPOIL, TIME, conc)
    else:
        a, x1, x2, q = mul_regr(TTOPOIL, TIME, conc)
    n = len(x1)
    yy = []
    if exp:
        for i in range(n):
            yy.append(a[0] * (math.e ** (a[1] * x1[i] + a[2] * x2[i])))
    else:
        for i in range(n):
            yy.append(a[0] * (x1[i] ** a[1]) * (x2[i] ** a[2]))
    return yy
