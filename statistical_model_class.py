import pandas as pd
import numpy as np
import sklearn as linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import r2_score
from statsmodels.api import OLS
from statsmodels.api import add_constant
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('https://raw.githubusercontent.com/UVA-CS5010/semester-project/main/all_df_updated.csv')
NBA = data[data['League'] == 'NBA']
WNBA = data[data['League'] == 'WNBA']
df = [NBA, WNBA]


class linearModel:
    def __init__(self, df):
        self.stats = df
        # clean up the dataframe
        columns_to_drop = ['League', 'Team', 'Position', 'Age', 'Avg_Salary', 'salary_ratio', 'Salary_Rank']
        self.stats = self.stats.dropna()
        self.stats = self.stats.drop(columns_to_drop, axis=1)
        # seperate response and predictors
        self.salary = self.stats.iloc[:, -1]
        self.stats = self.stats.iloc[:, 4:-1]
        # Split the sample data into train and test data
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.stats,
                                                                                self.salary,
                                                                                test_size=0.2, random_state=4)
        # fit the linear model
        self.model = LinearRegression().fit(self.X_train, self.Y_train)

    def feature_selection(self, direction=None):
        sfs = SequentialFeatureSelector(self.model, n_features_to_select=3, direction=direction)
        self.sfs = sfs.fit(self.X_train, self.Y_train)
        # filter the columns by the selected features
        self.X_train_columns = self.X_train.columns[sfs.get_support()]
        self.X_test_columns = self.X_test.columns[sfs.get_support()]
        self.X_train = self.X_train[self.X_train_columns]
        self.X_test = self.X_test[self.X_test_columns]
        # fit the linear model
        self.model = LinearRegression().fit(self.X_train, self.Y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        # calculate the R squared value for the model
        self.r_squared = r2_score(self.Y_test, self.y_pred)

    def calc_r_squared_adjusted(self, r_squared):
        num_pred = len(self.X_train)
        pred = len(self.X_train.columns)
        r_squared = abs(r_squared)
        r_squared_adjusted = 1 - ((1 - r_squared) * (num_pred - 1) / (num_pred - pred - 1))
        return r_squared_adjusted


for league in df:
    name = [x for x in globals() if globals()[x] is league][0]
    print("======" + str(name) + "======")
    full_model = linearModel(league)
    full_model.predict()
    print('Full model R Squared Valued: ' + str(full_model.r_squared))

    forward_model = linearModel(league)
    forward_model.feature_selection('forward')
    forward_model.predict()
    print('Forward model R Squared value: ' + str(forward_model.r_squared))

    backward_model = linearModel(league)
    backward_model.feature_selection('backward')
    backward_model.predict()
    print('Backward model R Squared value: ' + str(backward_model.r_squared))

    # Pick the best model based on the R squared value
    models = {full_model: full_model.r_squared, forward_model: forward_model.r_squared,
              backward_model: backward_model.r_squared}
    best = max(models, key=models.get)
    best_name = [x for x in globals() if globals()[x] is best][0]
    print('Based on the R Squared metric, the ' + str(best_name) + ' is the best choice.')
    # Print out the statistical summary for the best model
    best.X_train = add_constant(best.X_train)
    new_model = OLS(best.Y_train, best.X_train).fit()
    print(new_model.summary())
    # Plot the correlation matrix between all the predictors
    correlation = best.stats.corr()
    sns.heatmap(correlation, annot=True)
    plt.show()
    print("==============")
