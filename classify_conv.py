import pandas as pd

def classify_convert(x):
    if 0 <= x <= 80:
        return 1
    if 80 < x <= 160:
        return 2
    if 160 < x <= 240:
        return 3
    if 240 < x <= 320:
        return 4
    if x > 320:
        return 5

def frame_manip():
    # load data
    df = pd.read_csv('COsnowtotals.csv')
    # drop first column
    df = df.iloc[:, 1:]
    # print max and min for snow
    # print(df.snow.max())
    # print(df.snow.min())
    df['FIPS'] = df['FIPS'].astype(float)
    # 0-300 every 60 == class 5 classes 1-6 6 = op
    df['snow'] = df.snow.apply(lambda x: classify_convert(x))

    # get the X attributes and Y values
    df_x = df.iloc[:, :-1]
    df_y = df.iloc[:, -1]

    count = 26
    df_a = df_x.iloc[:count, :]
    df_b = df_y.iloc[:count]

    r = int(df_x.shape[0] / 26)

    for i in range(r-1):
        df_n = df_x.iloc[count:(count+26), :].reset_index(drop=True)
        df_m = df_y.iloc[count:(count+26)].reset_index(drop=True)
        count += 26
        df_a = pd.concat([df_a, df_n], axis=1)
        df_b = pd.concat([df_b, df_m], axis=1)

    df = pd.concat([df_a, df_b], axis=1)
    df.to_csv('test.csv')
