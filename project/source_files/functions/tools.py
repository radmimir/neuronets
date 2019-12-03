import xlrd
import statistics
from scipy import ndimage


def read_xl(sheet):
    a = xlrd.open_workbook("dataset_last_changed.xlsx")
    sheet = a.sheet_by_name(sheet)
    last = sheet.nrows
    print(last)
    first_ind = 0
    CO2 = sheet.col_values(0, first_ind, last)
    TTOPOIL = sheet.col_values(1, first_ind, last)
    TIME = sheet.col_values(2, first_ind, last)
    TTOPOIL = [i + 273 for i in TTOPOIL]
    TIME = [i - TIME[0] + 1 for i in TIME]
    n = len(CO2)
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
    return TIME, CO2, TTOPOIL, n


def gauss(y):
    sigma = statistics.stdev(y)
    return ndimage.gaussian_filter(y, sigma=sigma, order=0)
