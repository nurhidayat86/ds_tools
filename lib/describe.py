import pandas as pd
from sklearn.metrics import auc
from lib.base_proc import calculate_gini

def predictor_power(data, col_features, col_target):

    if isinstance(col_target, list):
        df_power = pd.DataFrame(columns=col_target, index=col_features)
        for i in range(0, len(col_target)):
            for j in range(0, len(col_features)):
                data = data.sort_values(by=[col_features[j]])
                gini = calculate_gini(data[col_features[j]], data[col_target[i]])
                print(f'{col_features[j]}: {gini}')
                df_power.loc[col_features[j], col_target[i]] = gini
    else:
        df_power = pd.DataFrame(index=col_features)
        for j in range(0, len(col_features)):
            data = data.sort_values(by=[col_features[j]])
            gini = 2 * (auc(data[col_features[j]], data[col_target])) - 1
            # print(f'{col_features[j]}: {gini}')
            df_power.loc[col_features[j], col_target] = gini
    return df_power