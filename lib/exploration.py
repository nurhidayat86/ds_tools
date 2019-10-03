import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def nan_share(data, col_display, col_time, step_xticks=3):
    temp = data.groupby(col_time).apply(lambda x: x.isna().sum()/(x.count()+x.isna().sum()))
    n = len(col_display)
    for i in range(0, n):
        plt.figure()
        plt.title(col_display[i])
        plt.plot(temp.index, temp[col_display[i]])
        plt.xticks(np.arange(temp.index[0], temp.index[-1]+1,step_xticks), rotation='vertical')
    plt.show()

