import pandas as pd
import classify_conv
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression


# MultiClass Multi Label classifier using a random forest and sklearn MultiOutputclassifier
def multiclass_multilable(n_e):
    # todo 33 to 238
    df, fips = classify_conv.frame_manip()
    # 33 due to 33 regions
    df_r = df.iloc[:, :-238].values
    df_r = pd.DataFrame(df_r)
    # get x columns
    df_x = df.iloc[:, :-238]
    # get y columns
    df_y = df.iloc[:, -238:]

    scaler = StandardScaler()
    df_x = scaler.fit_transform(df_x)

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=.20)

    # try 10000 82%
    forest = RandomForestClassifier(n_estimators=n_e)
    multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
    multi_target_forest.fit(x_train, y_train)

    y_pred = multi_target_forest.predict(x_test)
    y_pred = pd.DataFrame(y_pred)

    # todo 32 to 237
    count = -237
    count_two = 1

    y_p = y_pred.iloc[:, :count].copy()
    y_p.rename(columns={y_p.columns[0]: 'Y'}, inplace=True)
    y_t = y_test.iloc[:, :count].copy()
    y_t.rename(columns={y_t.columns[0]: 'Y'}, inplace=True)
    # todo 31 to 236
    for i in range(236):
        count += 1
        df_n = y_pred.iloc[:, count_two:count].reset_index(drop=True)
        df_m = y_test.iloc[:, count_two:count].reset_index(drop=True)
        df_n.rename(columns={df_n.columns[0]: 'Y'}, inplace=True)
        df_m.rename(columns={df_m.columns[0]: 'Y'}, inplace=True)
        count_two += 1
        y_p = y_p.append(df_n)
        y_t = y_t.append(df_m)

    a = y_p.values
    b = y_t.values
    tp, fp = classify_conv.tp_fp(a, b)

    accuracy = accuracy_score(y_t, y_p)*100

    print('MultiClass Multi-label Model Accuracy:{:.2f}%'.format(accuracy))
    f = pd.DataFrame(fips)
    p = pd.DataFrame(y_pred.iloc[0:1, :].T)

    f_p = pd.concat([f, p], axis=1)

    return tp, fp, accuracy, f_p, multi_target_forest, df_r

def multiclass_multilable_km():

    df, fips = classify_conv.frame_manip()
    # get x columns todo 238
    df_x = df.iloc[:, :-238]
    # get y columns todo 238
    df_y = df.iloc[:, -238:]

    scaler = StandardScaler()
    df_x = scaler.fit_transform(df_x)

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=.20)

    # try 10000 82%
    neigh = KNeighborsClassifier(n_neighbors=20)
    neigh.fit(x_train, y_train)

    y_pred = neigh.predict(x_test)
    y_pred = pd.DataFrame(y_pred)
    # todo 237
    count = -237
    count_two = 1

    y_p = y_pred.iloc[:, :count].copy()
    y_p.rename(columns={y_p.columns[0]: 'Y'}, inplace=True)
    y_t = y_test.iloc[:, :count].copy()
    y_t.rename(columns={y_t.columns[0]: 'Y'}, inplace=True)
    # todo 236
    for i in range(236):
        count += 1
        df_n = y_pred.iloc[:, count_two:count].reset_index(drop=True)
        df_m = y_test.iloc[:, count_two:count].reset_index(drop=True)
        df_n.rename(columns={df_n.columns[0]: 'Y'}, inplace=True)
        df_m.rename(columns={df_m.columns[0]: 'Y'}, inplace=True)
        count_two += 1
        y_p = y_p.append(df_n)
        y_t = y_t.append(df_m)

    a = y_p.values
    b = y_t.values
    tp, fp = classify_conv.tp_fp(a, b)

    accuracy = (tp/(tp+fp))*100

    print('KMeans Model Accuracy:{:.2f}%'.format(accuracy))
    p = pd.DataFrame(y_pred.iloc[0:1, :].T)
    return tp, fp, accuracy, p, neigh


def year_reg(df_r):
    # get x columns
    df_x = df_r.iloc[:-1, :]
    # get y columns
    df_y = df_r.iloc[-1:, :]

    x_train, x_test, y_train, y_test = train_test_split(df_x.T, df_y.T, test_size=.20)
    lr = LinearRegression()
    prediction = lr.fit(x_train, y_train).predict(df_x.T)
    prediction = pd.DataFrame(prediction.T)
    return prediction
