import tensorflow as tf
import matplotlib as mtl
import xlrd


# default_graph = tf.get_default_graph()

# c1 = tf.constant(1.0)

# second_graph = tf.Graph()
# with second_graph.as_default():
#    c2 = tf.constant(101.0)

# print(c2.graph is second_graph, c1.graph is second_graph)
# print(c2.graph is default_graph, c1.graph is default_graph)


def main():
    a = xlrd.open_workbook("dataset.xlsx")

    for sheetnum in range(0,11):
        sheet = a.sheet_by_index(0)
        labels = sheet.row_values(0)
        dick = dict.fromkeys(labels)
        for d in dick:
            dick[d] = []
        for rownum in range(1,sheet.nrows):
            for l in labels:
                dick[l].append(sheet.row_values(rownum))
    print(dick['TS'])
    # for rownum in range(sheet.nrows):
    #     row = sheet.row_values(rownum)
    #     a = row
    #     print(a)
    #     for c_el in row:
    #         print(c_el,end=' ')
    pass


if __name__ == '__main__':
    main()
