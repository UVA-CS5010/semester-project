import pandas as pd
#The following lines are to fix console printout on PyCharm
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 33)
import numpy as np
import sklearn as linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import r2_score, mean_squared_error

nba_data = pd.read_csv('/Users/Jakekolessar/Downloads/nba_df.csv')
nba_data = nba_data.drop(['Rk'], axis=1)
wnba_data = pd.read_csv('/Users/Jakekolessar/Downloads/wnba_df.csv')


class linearModel:
    # I think most of this can be combined if there is no return
    def __init__(self, df):
        self.stats = df

    def prepare_data(self):
        # import pdb; pdb.set_trace()
        self.stats = self.stats.dropna()
        self.stats = self.stats.iloc[:, 8:]
        self.stats = self.stats.drop(['average salary'], axis=1)
        return self.stats

    def train_test_split(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.stats.iloc[:, :22],
                                                                                self.stats['salary_float'],
                                                                                test_size=0.2, random_state=4)

    def create_model(self):
        self.model = LinearRegression().fit(self.X_train, self.Y_train)

    def feature_selection(self, direction=None):
        sfs = SequentialFeatureSelector(self.model, n_features_to_select=5, direction=direction)
        self.sfs = sfs.fit(self.X_train, self.Y_train)
        self.X_train_columns = self.X_train.columns[sfs.get_support()]
        self.X_test_columns = self.X_test.columns[sfs.get_support()]
        self.X_train = self.X_train[self.X_train_columns]
        self.X_test = self.X_test[self.X_test_columns]
        self.model = LinearRegression().fit(self.X_train, self.Y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)

    def calc_r_squared(self):
        self.r_squared = r2_score(self.Y_test, self.y_pred)

    def calc_r_squared_adjusted(self, r_squared):
        num_pred = len(self.X_train)
        pred = len(self.X_train.columns)
        r_squared = abs(r_squared)
        import pdb;
        pdb.set_trace()
        r_squared_adjusted = 1 - ((1 - r_squared) * (num_pred - 1) / (num_pred - pred - 1))
        return r_squared_adjusted


model = linearModel(nba_data)
model.prepare_data()
model.train_test_split()
model.create_model()
model.predict()
model.calc_r_squared()
print('Full model R Squared Valued: ' + str(model.r_squared))

forward_model = linearModel(nba_data)
forward_model.prepare_data()
forward_model.train_test_split()
forward_model.create_model()
forward_model.feature_selection('forward')
forward_model.predict()
forward_model.calc_r_squared()
print('Forward model R Squared value: ' + str(forward_model.r_squared))

backward_model = linearModel(nba_data)
backward_model.prepare_data()
backward_model.train_test_split()
backward_model.create_model()
backward_model.feature_selection('backward')
backward_model.predict()
backward_model.calc_r_squared()
print('Backward model R Squared value: ' + str(backward_model.r_squared))

models = {'full model': model.r_squared, 'forward_model': forward_model.r_squared,
          'backward_model': backward_model.r_squared}
best = max(models, key=models.get)
print('Based on the R Squared metric, the ' + str(best) + ' is the best choice.')
