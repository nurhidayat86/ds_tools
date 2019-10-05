import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def nan_share(data, col_display, col_time, step_xticks=3):
    temp = data.groupby(col_time).apply(lambda x: x.isna().sum()/(x.count()+x.isna().sum()))
    n = len(col_display)
    for i in range(0, n):
        plt.figure()
        plt.title(col_display[i])
        plt.plot(temp.index, temp[col_display[i]])
        plt.xticks(rotation='vertical')
        # plt.xticks(np.arange(temp.index[0], temp.index[-1]+1,step_xticks), rotation='vertical')
    plt.show()

def nan_mat(data, col_time):
    temp = data.groupby(col_time).apply(lambda x: x.isna().sum() / (x.count() + x.isna().sum()))
    print(temp.shape)
    plt.figure()
    plt.imshow(temp.transpose(), vmin=0, vmax=1, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    plt.yticks(np.arange(0, len(temp.columns)), temp.columns)
    plt.ylabel('Predictors')
    plt.xticks(np.arange(0, len(temp.index)), temp.index, rotation='vertical')
    plt.xlabel(f'Time ({col_time})')
    plt.title('NAN Share per predictor')
    plt.show()

def mat_default_rate(data, col_target, col_time):
    temp = data[[col_target, col_time]].groupby(col_time).apply(lambda x: x.sum() / x.count())
    temp2 = data[[col_target, col_time]].groupby(col_time).apply(lambda x: x.count())
    print(temp)
    plt.figure()
    plt.plot(temp.index, temp[col_target])
    plt.ylabel('Predictors')
    plt.xticks(np.arange(0, len(temp.index)), temp.index, rotation='vertical')
    plt.xlabel(f'Time ({col_time})')
    plt.title('Default rate')
    plt.show()