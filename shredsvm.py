import classify_conv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


def shred_svm():
    # load data
    df = pd.read_csv('COsnowtotals.csv')
    # drop first column
    df = df.iloc[:, 1:]
    # drop any rows with a snow total under 30
    df = df[df.snow > 30]

    # print max and min for snow
    #print(df.snow.max())
    #print(df.snow.min())

    # 0-300 every 60 == class 5 classes 1-6 6 = op
    df['snow'] = df.snow.apply(lambda x: classify_conv.classify_convert(x))

    # get the X attributes and Y values
    df_train = df.iloc[:, :-1]
    df_test = df.iloc[:, -1]

    # attempt 'to normalize' data only used on the X not y values
    scaler = StandardScaler()
    # scaler
    df_train = scaler.fit_transform(df_train)

    # split data for testing into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(df_train, df_test, test_size=.20, random_state=43)

    # Set classifer to one vs rest classifier using radial basis function
    clf = OneVsRestClassifier(SVC(kernel='rbf', cache_size=1000))

    # C grid search param
    C_range = [.001, .01, .1, 1, 10, 100, 1000]
    # gamma grid search param
    GAMMA_range = [.001, .01, .1, 1, 10, 100, 1000]
    param_grid = {'estimator__C': C_range,
                  'estimator__gamma': GAMMA_range}
    # use gridsearch to attempt to find best SVC parameters.
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    grid_search.fit(x_train, y_train)
    print('Best C Params: ', grid_search.best_params_)

    # get predictions
    y_pred = grid_search.predict(x_test)

    # print model accuracy.
    print("SVM Model Accuracy: {:.2f}%".format(grid_search.score(x_test, y_test) * 100))