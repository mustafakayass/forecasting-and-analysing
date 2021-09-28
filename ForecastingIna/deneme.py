import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


def import_data():
    data = pd.read_csv('INA.csv', sep=';')
    data['Period'] = data['Year'].astype(str) + '-' + data['Month'].astype(str).str.zfill(2)
    data['Period'] = pd.to_datetime(data['Period']).dt.strftime('% Y - % m')
    df = pd.pivot_table(data=data, values='Quantity', index='Material', columns='Period', aggfunc='sum',
                        fill_value=0)

    return df


def datasets(df, x_len=7, y_len=1, test_loops=0):
    D = df.values
    rows, periods = D.shape
    # training set creation
    loops = periods + 1 - x_len - y_len - test_loops
    train = []
    for col in range(loops):
        train.append(D[:, col:col + x_len + y_len])
    train = np.vstack(train)
    X_train, Y_train = np.split(train, [-y_len], axis=1)

    if test_loops > 0:
        X_train, X_test = np.split(X_train, [-rows * test_loops], axis=0)
        Y_train, Y_test = np.split(Y_train, [-rows * test_loops], axis=0)
    else:
        X_test = D[:, -x_len:]
        X_test = np.full((X_test.shape[0], y_len), np.nan)

    if y_len == 1:
        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()

    return X_train, Y_train, X_test, Y_test


df1 = import_data()

X_train, Y_train, X_test, Y_test = datasets(df1, x_len=4, y_len=1, test_loops=1)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg = reg.fit(X_train, Y_train)

Y_train_pred = reg.predict(X_train)
Y_test_pred = reg.predict(X_test)


def kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name=''):
    df = pd.DataFrame(columns=['MAE', 'RMSE', 'Bias'], index=['Train', 'Test'])
    df.index.name = name
    df.loc['Train', 'MAE'] = 100 * np.mean(abs(Y_train - Y_train_pred)) / np.mean(Y_train)
    df.loc['Train', 'RMSE'] = 100 * np.sqrt(np.mean((Y_train - Y_train_pred) ** 2)) / np.mean(Y_train)
    df.loc['Train', 'Bias'] = 100 * np.mean((Y_train - Y_train_pred)) / np.mean(Y_train)
    df.loc['Test', 'MAE'] = 100 * np.mean(abs(Y_test - Y_test_pred)) / np.mean(Y_test)
    df.loc['Test', 'RMSE'] = 100 * np.sqrt(np.mean((Y_test - Y_test_pred) ** 2)) / np.mean(Y_test)
    df.loc['Test', 'Bias'] = 100 * np.mean((Y_test - Y_test_pred)) / np.mean(Y_test)
    df = df.astype(float).round(1)
    print(df)


def seasonal_factors_mul(s, d, slen, cols):
    for i in range(slen):
        s[i] = np.mean(d[i:cols:slen])
    s /= np.mean(s[:slen])
    return s


def triple_exp_smooth_mul(d, slen=12, extra_periods=1, alpha=0.4,
                          beta=0.4, phi=0.9, gamma=0.3):
    cols = len(d)
    d = np.append(d, [np.nan] * extra_periods)
    f, a, b, s = np.full((4, cols + extra_periods), np.nan)
    s = seasonal_factors_mul(s, d, slen, cols)
    a[0] = d[0] / s[0]
    b[0] = d[1] / s[1] - d[0] / s[0]

    for t in range(1, slen):
        f[t] = (a - [t - 1] + phi * b[t - 1]) * s[t]
        a[t] = alpha * d[t] / s[t] + (1 - alpha) * (a[t - 1] + phi * b[t - 1])
        b[t] = beta * (a[t] - a[t - 1]) + (1 - beta) * phi * b[t - 1]

    for t in range(slen, cols):
        f[t] = (a[t - 1] + phi * b[t - 1]) * s[t - slen]
        a[t] = alpha * d[t] / s[t - slen] + (1 - alpha) * (a[t - 1] + phi * b[t - 1])
        b[t] = beta * (a[t] - a[t - 1]) + (1 - beta) * phi * b[t - 1]
        s[t] = gamma * d[t] / a[t] + (1 - gamma) * s[t - slen]
    for t in range(cols, cols + extra_periods):
        f[t] = (a[t - 1] + phi * b[t - 1]) * s[t - slen]
        a[t] = f[t] / s[t - slen]
        b[t] = phi * b[t - 1]
        s[t] = s[t - slen]
    dframe = pd.DataFrame.from_dict({'Demand': d, 'Forecast': f, 'Level': a,
                                     'Trend': b, 'Season': s, 'Error': d - f})
    return dframe


kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='Regression')

X_train, Y_train, X_test, Y_test = datasets(df1, x_len=2, y_len=3, test_loops=1)
reg = LinearRegression()
reg = reg.fit(X_train, Y_train)
forecast = pd.DataFrame(data=reg.predict(X_test), index=df1.index)
print(forecast)

'''
x = int(input('Hangi materyal tahmini istiyorsunuz?: \n'))
print(forecast.loc[[x]])
'''

forecast.to_csv('forecast.csv', index=True)

'''df1.plot(secondary_y=['Season'])
plt.show()

plt.plot(forecast, color='g')
plt.show()
'''