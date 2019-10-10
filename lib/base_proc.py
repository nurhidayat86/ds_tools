from sklearn.metrics import auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def corr_mat(data, col_features, method='pearson', min_periods=1):
    return data[col_features].corr(method=method, min_periods=min_periods)

def calculate_gini(y_pred, y_true):
    return (2*auc(y_pred, y_true)-1)

def produce_lift(data, col_score, col_target='y', n_bin = 100):
    df = data[[col_score, col_target]].copy()
    df = df.sort_values(by=col_score, ascending=True)
    df = df.reset_index()
    bin = pd.cut(df.index, n_bin, labels=np.arange(1,n_bin+1))
    df['bin'] = [f'{i}' for i in bin]
    total_target = df[col_target].sum()
    print(total_target)
    lift_mat = pd.DataFrame(columns=['lift','ideal'], index=df['bin'].unique())
    lift_mat['ideal'] = 1

    for i in lift_mat.index:
        lift_mat.loc[i, 'lift'] = (100*df[col_target].iloc[0:i+1].sum())/(total_target*(i+1))

    # plt.figure()
    # plt.plot(list(lift_mat.index), list(lift_mat['lift']))
    # plt.plot(list(lift_mat.index), list(lift_mat['ideal']))
    # plt.show()

    return lift_mat

def produce_KS(data, col_score, col_target='y', n_bin = 10):
    df = data[[col_score, col_target]].copy()
    df = df.sort_values(by=col_score, ascending=True)
    df['bin'] = pd.cut(df[col_score], n_bin)
    temp = df[['bin', col_target]].groupby('bin').apply(lambda x: x.loc[x[col_target]==0, col_target].count())
    temp2 = df[['bin', col_target]].groupby('bin').apply(lambda x: x.loc[x[col_target]==1, col_target].count())
    temp = pd.concat([temp, temp2], axis=1)
    total_1 = temp[1].sum()
    total_0 = temp[0].sum()
    temp['cum_0'] = 0
    temp['cum_1'] = 0

    for i in range(0, temp.shape[0]):
        temp.loc[temp.index[i], 'cum_0'] = 100*temp[0].iloc[0:i+1].sum() / total_0
        temp.loc[temp.index[i], 'cum_1'] = 100*temp[1].iloc[0:i + 1].sum() / total_1

    temp['ks'] = (temp['cum_0'] - temp['cum_1']).abs()

    # plt.figure()
    # plt.plot([f'{i}' for i in temp.index], temp['cum_0'])
    # plt.plot([f'{i}' for i in temp.index], temp['cum_1'])
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.show()

    print(temp)
    return temp



