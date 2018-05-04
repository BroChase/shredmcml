import mcml
import shredsvm
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    tp_mc, fp_mc, mc_acc = mcml.multiclass_multilable(n_e=10)
    tp_svm, fp_svm, svm_acc = shredsvm.shred_svm()
    tp_km, fp_km, km_acc = mcml.multiclass_multilable_km()

    N = 4

    ind = np.arange(N)

    width = 0.25

    fp = 'orangered'
    tp = 'cadetblue'

    p1 = plt.bar(1, fp_mc, width, color=fp)
    p2 = plt.bar(1, tp_mc, width, bottom=fp_mc, color=tp)
    p3 = plt.bar(2, fp_svm, width, color=fp)
    p4 = plt.bar(2, tp_svm, width, bottom=fp_svm, color=tp)
    p5 = plt.bar(3, fp_km, width, color=fp)
    p6 = plt.bar(3, tp_km, width, bottom=fp_km, color=tp)


    for rect in p2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height,
                 '{:.2f}%'.format(mc_acc), ha='center')
    for rect in p4:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height,
                 '{:.2f}%'.format(svm_acc), ha='center', va='top')
    for rect in p6:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height,
                 '{:.2f}%'.format(km_acc), ha='center', va='top')

    plt.ylabel('Samples')
    plt.title('Model Results')
    plt.xticks(ind, ('0', 'MCML(forest)', 'SVM', 'KMeans'))
    plt.legend((p1, p2), ('Miss_Classified', 'Correctly_Classified'))
    plt.show()

