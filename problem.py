import os
import pandas as pd
import rampwf as rw

# TODO add workflow : https://github.com/paris-saclay-cds/ramp-workflow/wiki/Build-your-own-workflow

problem_title = 'Nutriscore prediction'

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, f_name), sep='|', compression='zip')
    y_array = data['nutriscore_grade'].values
    X_df = data.drop(['nutriscore_grade'], axis=1)
    return X_df, y_array

def get_train_data(path='.'):
    f_name = 'data_train.csv.zip'
    return _read_data(path, f_name)

def get_test_data(path='.'):
    f_name = 'data_test.csv.zip'
    return _read_data(path, f_name)
