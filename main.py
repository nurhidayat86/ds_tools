from lib.data_generator import generate_linear, add_multicolinear, add_timestamp, nullize_columns, add_timeflag
from lib.exploration import show_monotonic_percentile
import pandas as pd
import gc

if __name__ == '__main__':
    data = generate_linear(n_data=100, n_cat=3, n_num=5, cat_max=5, num_min=0, num_max=100, max_weight=1, noise=0.3,
                           logistic=True)
    data = add_multicolinear(data,['num_1'])
    data = add_timestamp(data)
    data = nullize_columns(data, data.columns[3:6], ratio=[0.2, 0.4, 0.5])
    data = add_timeflag(data, 'TIME')
    print(data.loc[data['MONTH']==201801,'num_2'])
    # nan_share(data, data.columns[3:6], 'DAY')
    # nan_mat(data, 'MONTH')
    show_monotonic_percentile(data, 'y', 20)
    # print(data)
    del data
    gc.collect()