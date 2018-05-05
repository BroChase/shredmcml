import pandas as pd

def classify_convert(x):
    if 0 <= x <= 50:
        return 1
    if 50 < x <= 100:
        return 2
    if 100 < x <= 150:
        return 3
    if 150 < x <= 320:
        return 4
    if x > 320:
        return 5

def tp_fp(y_t, y_p):
    tp = 0
    fp = 0
    for i in range(0, len(y_t)):
        if y_p[i] == y_t[i]:
            tp += 1
        elif y_p[i] != y_t[i]:
            fp += 1
    return tp, fp

def frame_manip():
    # load data
    df = pd.read_csv('COsnowtotals.csv')
    #df = pd.read_csv('snowtotals.csv')
    # drop first column
    df = df.iloc[:, 1:]
    # print max and min for snow
    # print(df.snow.max())
    # print(df.snow.min())
    fips = []
    f = 0
    div = int(df.shape[0]/33)
    for i in range(33):
        num = df.iloc[f:f+1, 0:1].values
        fips.append(num[0][0])
        f += div
    df['FIPS'] = df['FIPS'].astype(float)
    # 0-300 every 60 == class 5 classes 1-6 6 = op
    df['snow'] = df.snow.apply(lambda x: classify_convert(x))

    # get the X attributes and Y values
    df_x = df.iloc[:, :-1]
    df_y = df.iloc[:, -1]

    count = div
    df_a = df_x.iloc[:count, :]
    df_b = df_y.iloc[:count]

    r = int(df_x.shape[0] / div)

    for i in range(r-1):
        df_n = df_x.iloc[count:(count+div), :].reset_index(drop=True)
        df_m = df_y.iloc[count:(count+div)].reset_index(drop=True)
        count += div
        df_a = pd.concat([df_a, df_n], axis=1)
        df_b = pd.concat([df_b, df_m], axis=1)

    df = pd.concat([df_a, df_b], axis=1)
    # df.to_csv('test.csv')
    return df, fips

def frame_manip_single_year():
    # load data
    df = pd.read_csv('snowtotals.csv')
    #df = pd.read_csv('snowtotals.csv')
    # drop first column
    df = df.iloc[:, 1:]
    div = int(df.shape[0]/33)
    df['FIPS'] = df['FIPS'].astype(float)
    # 0-300 every 60 == class 5 classes 1-6 6 = op
    df['snow'] = df.snow.apply(lambda x: classify_convert(x))

    # reorder the columns
    #df = df.reindex(columns=sorted(df.columns))
    #df = df.reindex(columns=(['FIPS'] + list([a for a in df.columns if a != 'FIPS'])))
    # get the X attributes and Y values
    df_x = df.iloc[:, :-1]
    df_y = df.iloc[:, -1]

    count = div
    df_a = df_x.iloc[:count, :]
    df_b = df_y.iloc[:count]

    r = int(df_x.shape[0] / div)

    for i in range(r-1):
        df_n = df_x.iloc[count:(count+div), :].reset_index(drop=True)
        df_m = df_y.iloc[count:(count+div)].reset_index(drop=True)
        count += div
        df_a = pd.concat([df_a, df_n], axis=1)
        df_b = pd.concat([df_b, df_m], axis=1)

    df = pd.concat([df_a, df_b], axis=1)
    df = df.iloc[:, :].values
    df = pd.DataFrame(df)
    return df

def single_year():
    df = pd.read_csv('snowtotals.csv')
    #df = pd.read_csv('snowtotals.csv')
    # drop first column
    df = df.iloc[:, 1:]
    div = int(df.shape[0]/33)
    df['FIPS'] = df['FIPS'].astype(float)
    # 0-300 every 60 == class 5 classes 1-6 6 = op
    df['snow'] = df.snow.apply(lambda x: classify_convert(x))
    df = df.iloc[:, :-1]

    return df