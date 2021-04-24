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
from statsmodels.api import OLS
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('https://raw.githubusercontent.com/UVA-CS5010/semester-project/main/NBAvsWNBA.csv')
nba_data = data[data['League']=='NBA']
wnba_data = data[data['League']=='WNBA']

class linearModel:
    def __init__(self, df):
        self.stats = df
        #lean up the dataframe
        columns_to_drop = ['League', 'Team', 'Position', 'Age', 'Avg_Salary', 'salary_ratio', 'Salary_Rank']
        self.stats = self.stats.dropna()
        self.stats = self.stats.drop(columns_to_drop, axis=1)
        #seperate response and predictors
        self.salary = self.stats.iloc[:, -1]
        self.stats = self.stats.iloc[:, 4:-1]
        # Split the sample data into train and test data
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.stats,
                                                                                self.salary,
                                                                                test_size=0.2, random_state=4)
        #fit the linear model
        self.model = LinearRegression().fit(self.X_train, self.Y_train)

    def feature_selection(self, direction=None):
        sfs = SequentialFeatureSelector(self.model, n_features_to_select=3, direction=direction)
        self.sfs = sfs.fit(self.X_train, self.Y_train)
        self.X_train_columns = self.X_train.columns[sfs.get_support()]
        self.X_test_columns = self.X_test.columns[sfs.get_support()]
        self.X_train = self.X_train[self.X_train_columns]
        self.X_test = self.X_test[self.X_test_columns]
        self.model = LinearRegression().fit(self.X_train, self.Y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        #calculate the R squared value for the model
        self.r_squared = r2_score(self.Y_test, self.y_pred)

    def calc_r_squared_adjusted(self, r_squared):
        num_pred = len(self.X_train)
        pred = len(self.X_train.columns)
        r_squared = abs(r_squared)
        r_squared_adjusted = 1 - ((1 - r_squared) * (num_pred - 1) / (num_pred - pred - 1))
        return r_squared_adjusted


model = linearModel(nba_data)
model.predict()
print('Full model R Squared Valued: ' + str(model.r_squared))

forward_model = linearModel(nba_data)
forward_model.feature_selection('forward')
forward_model.predict()
print('Forward model R Squared value: ' + str(forward_model.r_squared))

backward_model = linearModel(nba_data)
backward_model.feature_selection('backward')
backward_model.predict()
print('Backward model R Squared value: ' + str(backward_model.r_squared))

# Pick the best model based on the R squared value
models = {'full model': model.r_squared, 'forward_model': forward_model.r_squared,
          'backward_model': backward_model.r_squared}
best = max(models, key=models.get)
print('Based on the R Squared metric, the ' + str(best) + ' is the best choice.')

# Print out the statistical summary for the best model
new_model = OLS(forward_model.Y_train, forward_model.X_train).fit()
print(new_model.summary())