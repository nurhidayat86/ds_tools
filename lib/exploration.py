import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()

def nan_share(data, col_display, col_time, step_xticks=3):
    temp = data.groupby(col_time).apply(lambda x: x.isna().sum()/(x.count()+x.isna().sum()))
    n = len(col_display)
    for i in range(0, n):
        plt.figure()
        plt.title(col_display[i])
        plt.plot(temp.index, temp[col_display[i]])
        plt.xticks(rotation='vertical')
        # plt.xticks(np.arange(temp.index[0], temp.index[-1]+1,step_xticks), rotation='vertical')
        plt.tight_layout()
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
    plt.tight_layout()
    plt.show()

def show_default_rate(data, col_target, col_time):
    temp = data[[col_target, col_time]].groupby(col_time).apply(lambda x: x.sum() / x.count())
    temp2 = data[[col_target, col_time]].groupby(col_time).apply(lambda x: x.count())
    temp2_yticks = np.linspace(0,temp2[col_target].max(),10).round()

    #Plotting with twinx
    fig, ax1 = plt.subplots()
    ax1.bar(temp2.index, temp2[col_target])
    ax1.set_xticks(np.arange(0, len(temp.index)))
    ax1.set_xticklabels(temp.index, rotation=90)
    ax1.set_xlabel(f'Time in {col_time}')
    ax1.set_yticks(temp2_yticks)
    ax1.set_yticklabels(temp2_yticks, c='blue')
    ax1.set_ylabel('# Samples', c='blue')
    ax2 = ax1.twinx()
    ax2.plot(temp.index, temp[col_target], c='red')
    ax2.set_yticks(np.arange(0,1.1,0.1))
    ax2.set_yticklabels([f'{i}0%' for i in range(0, 101)], color='red')
    ax2.set_ylabel('Default rate', c='red')
    # ax2.ylabel('Predictors')
    # ax2.xlabel(f'Time ({col_time})')
    print(temp2)
    fig.tight_layout()
    fig.show()

def show_monotonic(data, col_target, col_predictors, n_bin=10):
    for idx in range(0, len(col_predictors)):
        col_feature = col_predictors[idx]
        bin = pd.cut(data[col_feature], n_bin)
        data['bin'] = [f'{i}' for i in bin]
        temp = data[['bin', col_target]].groupby('bin').apply(lambda x: x.sum() / x.count())
        temp2 = data[['bin', col_target]].groupby('bin').apply(lambda x: x.count())
        temp2_yticks = np.linspace(0,temp2[col_target].max(),10).round()

        #Plotting with twinx
        fig, ax1 = plt.subplots()
        ax1.bar(temp2.index, temp2[col_target])
        ax1.set_xticks(np.arange(0, len(temp.index)))
        ax1.set_xticklabels(temp.index, rotation=90)
        ax1.set_xlabel(f'Binning of {col_feature}')
        ax1.set_yticks(temp2_yticks)
        ax1.set_yticklabels(temp2_yticks, c='blue')
        ax1.set_ylabel('# Samples', c='blue')
        ax2 = ax1.twinx()
        ax2.plot(temp.index, temp[col_target], c='red')
        ax2.set_yticks(np.arange(0,1.1,0.1))
        ax2.set_yticklabels([f'{i}0%' for i in range(0, 101)], color='red')
        ax2.set_ylabel('Default rate', c='red')
        # ax2.ylabel('Predictors')
        # ax2.xlabel(f'Time ({col_time})')
        fig.tight_layout()
        fig.show()

def show_monotonic_percentile(data, col_target, n_bin=10):
    data['bin'] = pd.cut(data.index, n_bin, labels=np.arange(1,n_bin+1)*10)
    temp = data[['bin', col_target]].groupby('bin').apply(lambda x: x.sum() / x.count())
    fig, ax2 = plt.subplots()
    ax2_ticks = np.linspace(temp.index.min(), temp.index.max(), n_bin)
    ax2_ticklabels = [f'{i}%' for i in np.linspace(100/n_bin, 100, n_bin)]
    ax2.plot(temp.index, temp[col_target], c='blue')
    ax2.set_xticks(ax2_ticks)
    ax2.set_xticklabels(ax2_ticklabels, rotation=90)
    ax2.set_xlabel(f'Percentile of n={n_bin}')
    ax2.set_yticks(np.arange(0,1.1,0.1))
    ax2.set_yticklabels([f'{i}0%' for i in range(0, 101)], color='blue')
    ax2.set_ylabel('Default rate', c='Blue')
    # ax2.ylabel('Predictors')
    # ax2.xlabel(f'Time ({col_time})')
    fig.tight_layout()
    fig.show()

def show_corr_mat(data, col_features, method='spearman', min_periods=1):
    cm = data[col_features].corr(method=method, min_periods=min_periods)
    # print(cm.columns)
    fig, ax = plt.subplots()
    im = ax.matshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    xticklabel = cm.columns.tolist()
    yticklabel = cm.index.tolist()
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
           # ... and label them with the respective list entries
           # xticks=np.arange(-1,len(cm.columns)+1), yticks=np.arange(-1,len(cm.index)+1),
           xticklabels=['']+xticklabel, yticklabels=['']+yticklabel,
           title=f'Correlation matrix: {method}')

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), ha="right",
    #          rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f'
    # thresh = cm.max() / 2.
    thresh = 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm.iloc[i, j], fmt),
                    ha="center", va="center",
                    # color="white")
                    color="white" if cm.iloc[i, j] > 0.5 or cm.iloc[i, j] < -0.5 else "black")
    fig.tight_layout()
    fig.show()
