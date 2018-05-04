import pandas as pd
import numpy as np
import classify_conv
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler


def multiclass_multilable():
    df = classify_conv.frame_manip()
    # drop first index column
    df = df.iloc[:, 1:]
    # get x columns
    df_x = df.iloc[:, :-33]
    # get y columns
    df_y = df.iloc[:, -33:]

    # scaler = StandardScaler()
    # df_x = scaler.fit_transform(df_x)

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=.20)

    # try 10000 82%
    forest = RandomForestClassifier(n_estimators=10)
    multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
    multi_target_forest.fit(x_train, y_train)

    y_pred = multi_target_forest.predict(x_test)
    y_pred = pd.DataFrame(y_pred)
    count = -32
    count_two = 1

    y_p = y_pred.iloc[:, :count]
    y_p.rename(columns={y_p.columns[0]: 'Y'}, inplace=True)
    y_t = y_test.iloc[:, :count]
    y_t.rename(columns={y_t.columns[0]: 'Y'}, inplace=True)

    for i in range(31):
        count += 1
        df_n = y_pred.iloc[:, count_two:count].reset_index(drop=True)
        df_m = y_test.iloc[:, count_two:count].reset_index(drop=True)
        df_n.rename(columns={df_n.columns[0]: 'Y'}, inplace=True)
        df_m.rename(columns={df_m.columns[0]: 'Y'}, inplace=True)
        count_two += 1
        y_p = y_p.append(df_n)
        y_t = y_t.append(df_m)

    n_classes = len(y_p)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        print(y_p.iloc[:])
        # fpr[i], tpr[i], _ = roc_curve(y_t.iloc[:, i],y_p.iloc[:,i])
        # roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


    print(accuracy_score(y_p, y_t))