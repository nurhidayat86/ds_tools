import numpy as np
import pandas as pd
import gc
from datetime import datetime

def nullize_columns(data, columns, ratio=0.4):
    for i in range(0, len(columns)):
        idx = list(data.index)
        np.random.shuffle(idx)
        if isinstance(ratio, list):
            idx = idx[0:int(ratio[i] * len(data))]
        else:
            idx = idx[0:int(ratio * len(data))]
        data.loc[idx,columns[i]] = None
    return data

def create_timestamp(rownum=100000, min_date='2018-01-01 00:00:00', max_date='2019-12-31 23:59:59', format='%Y-%m-%d %H:%M:%S'):
    min_date = datetime.strptime(min_date, format)
    max_date = datetime.strptime(max_date, format)
    min_date = (min_date - datetime(1970,1,1)).total_seconds()
    max_date = (max_date - datetime(1970,1,1)).total_seconds()
    data = np.linspace(min_date, max_date, rownum)
    data = [datetime.utcfromtimestamp(i) for i in data]
    return data

def generate_categorical(n_features=20, n_data=100000, cat_max=5):
    features = np.random.rand(n_data, n_features)
    columns = [f'cat_{i}' for i in range(0,n_features)]
    data = pd.DataFrame(features, columns=columns)
    for i in range(0, n_features):
        n_cat = np.random.randint(2,cat_max)
        num_data = n_data // n_cat
        data = data.sort_values(by=[columns[i]])
        for j in range(0, n_cat):
            if j<(n_cat-1):
                data.iloc[j*num_data:(j+1)*num_data,i]=f'{columns[i]}_no_{j}'
            else:
                data.iloc[j*num_data:,i] = f'{columns[i]}_{j}'
    return data

def generate_numerical(n_features=20, num_min=100, num_max=1000, n_data=100000):
    if isinstance(num_min, list) and isinstance(num_min, list):
        for i in range(0, len(num_min)):
            if i == 0:
                features = np.random.randint(num_min[i], num_max[i], size=(1, n_features))
            else:
                features = np.concatenate([features, np.random.randint(num_min[i], num_max[i], size=(1, n_features))],
                                          axis=0)
    else:
        features = np.random.randint(num_min, num_max, size=(n_data, n_features))
    columns = [f'num_{i}' for i in range(0, n_features)]
    return pd.DataFrame(features, columns=columns)

def generate_linear(n_data=100000, n_cat=10, n_num=10, cat_max=5, num_min=0, num_max=100, max_weight=100, noise=0.3,
                    logistic=False, threshold=0.7):
    df_num = generate_numerical(n_features=n_num, num_min=num_min, num_max=num_max, n_data=n_data)
    df_cat = generate_categorical(n_features=n_cat, n_data=n_data, cat_max=cat_max)
    df_dummy = pd.get_dummies(df_cat, drop_first=False)
    data = pd.concat([df_num, df_dummy], axis=1)
    weight = np.random.rand(data.shape[1]) * max_weight
    y = (np.matmul(data.values, weight) + np.random.rand(data.shape[0]) * noise * max_weight).reshape(data.shape[0],1)
    y = pd.DataFrame(y, columns=['y'])
    del data
    gc.collect()
    # data = pd.concat([y, df_num, df_cat], axis=1)
    data = pd.concat([pd.DataFrame(data=np.arange(0,n_data), columns=['IDX']), y, df_num, df_cat], axis=1)
    if logistic:
        data = data.sort_values(by=['y'])
        data = data.reset_index()
        threshold = int(threshold*n_data)
        data.loc[:, 'y'] = 1
        data.loc[0:threshold-1, 'y'] = 0
        data = data.sort_values(by=['IDX'])
        data.index = data['IDX']
        data = data.drop(['IDX', 'index'], axis=1)
    return data

def add_timestamp(data, min_date='2018-01-01 00:00:00', max_date='2019-12-31 23:59:59', format='%Y-%m-%d %H:%M:%S', column='TIME'):
    rownum = data.shape[0]
    timestamp = create_timestamp(rownum, min_date, max_date, format)
    data[column] = timestamp
    return data

def add_multicolinear(data, col, weight=1, noise=10):
    if isinstance(col, list):
        for i in col:
            temp = weight*data[i].values+(np.random.rand(data.shape[0]) * noise)
            data[f'multi_{i}'] = temp
    else:
        temp = weight * data[col].values + (np.random.rand(data.shape[0]) * noise)
        data[f'multi_{col}'] = temp
    return data

def add_timeflag(data, col_time, unit='both'):
    if unit == 'month':
        data['MONTH'] = list(map(int,data[col_time].dt.strftime('%Y%m')))
    elif unit == 'day':
        data['DAY'] = list(map(int,data[col_time].dt.strftime('%Y%m%d')))
    else:
        data['MONTH'] = list(map(int, data[col_time].dt.strftime('%Y%m')))
        data['DAY'] = list(map(int, data[col_time].dt.strftime('%Y%m%d')))
    return data
