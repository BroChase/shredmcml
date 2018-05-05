import mcml
import shredsvm
import classify_conv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':

    # true positives, false positives, model accuracy, predictions.
    tp_mc, fp_mc, mc_acc, f_p, forest_model = mcml.multiclass_multilable(n_e=100)
    tp_km, fp_km, km_acc, km_p, kmeans_model = mcml.multiclass_multilable_km()
    tp_svm, fp_svm, svm_acc, svm_p, svm_model = shredsvm.shred_svm()

    df = classify_conv.frame_manip_single_year()

    # get x columns
    df_x = df.iloc[:, :-33]
    # get y columns
    df_y = df.iloc[:, -33:]
    scaler = StandardScaler()
    df_x = scaler.fit_transform(df_x)

    y_pred_forest = forest_model.predict(df_x)

    y_pred_kmeans = kmeans_model.predict(df_x)

    svm_x = classify_conv.single_year()
    svm_x = scaler.fit_transform(svm_x)
    y_pred_svm = svm_model.predict(svm_x)

    f_p = f_p.iloc[:, :-1].values
    df = pd.DataFrame(f_p, columns=['FIPS'])

    df1 = pd.DataFrame(y_pred_forest.T, columns=['MCML'])

    df2 = pd.DataFrame(y_pred_kmeans.T, columns=['KMeans'])

    df3 = pd.DataFrame(y_pred_svm.T, columns=['SVM'])

    merged = pd.concat([df, df1, df2, df3], axis=1)
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

    sales = [('MCML', tp_mc, fp_mc, mc_acc),
             ('Kmeans', tp_km, fp_km, km_acc),
             ('SVM', tp_svm, fp_svm, svm_acc)]
    labels = ['Model', 'Correctly_clf', 'Miss_clf', 'Accuracy']
    df = pd.DataFrame.from_records(sales, columns=labels)
    df.to_csv('modelresults.csv', index=False)

