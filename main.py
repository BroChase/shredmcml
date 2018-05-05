import mcml
import shredsvm
import classify_conv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':


    classify_conv.frame_manip_single_year()
    # true positives, false positives, model accuracy, predictions.
    tp_mc, fp_mc, mc_acc, f_p = mcml.multiclass_multilable(n_e=1000)
    tp_svm, fp_svm, svm_acc, svm_p = shredsvm.shred_svm()
    tp_km, fp_km, km_acc, km_p = mcml.multiclass_multilable_km()




    f_p = f_p.iloc[:, :].values
    df = pd.DataFrame(f_p, columns=['FIPS', 'MCML'])
    svm_p = svm_p.iloc[:, :].values
    df2 = pd.DataFrame(svm_p, columns=['SVM'])
    km_p = km_p.iloc[:, :].values
    df3 = pd.DataFrame(km_p, columns=['KMeans'])
    merged = pd.concat([df, df2, df3], axis=1)
    merged.to_csv('results.csv', index=False)

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

