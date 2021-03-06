import pandas as pd

# used to classify what 'we' think is good snow totals
# for a skier
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


# clculate how many true classified/false classified
def tp_fp(y_t, y_p):
    tp = 0
    fp = 0
    for i in range(0, len(y_t)):
        if y_p[i] == y_t[i]:
            tp += 1
        elif y_p[i] != y_t[i]:
            fp += 1
    return tp, fp

# Manipulate the data into the frame needed for training specific models
def frame_manip():
    # load data
    df = pd.read_csv('snowtotals_multi.csv')
    #df = pd.read_csv('snowtotals.csv')
    # drop first column
    df = df.iloc[:, 1:]
    # print max and min for snow
    # print(df.snow.max())
    # print(df.snow.min())
    fips = []
    f = 0
    # todo 238
    div = int(df.shape[0]/238)
    for i in range(238):
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
    df = df.iloc[:, :].values
    df = pd.DataFrame(df)
    # df.to_csv('test.csv')
    return df, fips

def frame_manip_single_year():
    # load data
    df = pd.read_csv('snowtotals_2.csv')
    #df = pd.read_csv('snowtotals.csv')
    # drop first column
    df = df.iloc[:, 1:]
    # todo 238
    div = int(df.shape[0]/238)
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


# manipulate the dataframe that is passed in to format for use with the svm model
def single_year(df):
    df = pd.DataFrame(df)
    # div = number of features belonging to each FIPS todo 238
    div = int(df.shape[1]/238)
    count = 0
    di = div
    # create a new frame to append to
    new_df = df.iloc[:, count:div].values
    new_df = pd.DataFrame(new_df)
    # append all the frame creating a columnwise frame. todo 237
    for i in range(237):
        count += di
        div += di
        df_n = df.iloc[:, count:div].values
        df_n = pd.DataFrame(df_n)
        new_df = new_df.append(df_n, ignore_index=True)

    return new_df
