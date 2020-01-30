import os
import pandas as pd
import rampwf as rw
from rampwf.workflows import FeatureExtractorClassifier
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import GroupShuffleSplit
import numpy as np

problem_title = 'Nutriscore prediction'
_target_column_name = 'nutriscore_grade' 

_prediction_label_names = ['a', 'b', 'c', 'd', 'e']

Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)

class NSG(FeatureExtractorClassifier):
    def __init__(self, workflow_element_names=[
            'feature_extractor', 'classifier']):
        super(NSG, self).__init__(workflow_element_names[:2])
        self.element_names = workflow_element_names

workflow = NSG()

# define the score (specific score for the NSG problem)
class NSG_error(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='nsg error', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        if isinstance(y_true, pd.Series):
            y_true = y_true.values

        loss = np.mean(2*np.maximum(0, (y_true - y_pred)*np.sqrt(np.maximum(0, (y_true - y_pred)))) + 2*np.maximum(0, (y_pred - y_true)**2))

        return loss

score_types = [
    NSG_error(name='nsg error', precision=2),
]

def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, f_name), sep=',', compression='zip')
    y_array = data['nutriscore_grade'].values
    X_df = data.drop(['nutriscore_grade'], axis=1)
    return X_df, y_array

def get_train_data(path='./data/'):
    f_name = 'data_train.csv.zip'
    return _read_data(path, f_name)

def get_test_data(path='./data/'):
    f_name = 'data_test.csv.zip'
    return _read_data(path, f_name)

def get_cv(X, y):
    cv = GroupShuffleSplit(n_splits=8, test_size=0.20, random_state=42)
    return cv.split(X,y, groups=X['_id'])
