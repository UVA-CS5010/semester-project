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


data = pd.read_csv('https://raw.githubusercontent.com/UVA-CS5010/semester-project/main/NBAvsWNBA.csv')
NBA = data[data['League'] == 'NBA']
WNBA = data[data['League'] == 'WNBA']
df = [NBA, WNBA]


class linearModel:
    def __init__(self, df):
        self.stats = df
        # clean up the dataframe
        columns_to_drop = ['League', 'Team', 'Position', 'Age', 'Avg_Salary', 'salary_ratio', 'GS%', 'Salary_Rank']
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

# for league in df:
name = [x for x in globals() if globals()[x] is NBA][0]
print("======" + str(name) + "======")
NBA_full_model = linearModel(NBA)
NBA_full_model.predict()
print('Full model R Squared Valued: ' + str(NBA_full_model.r_squared))

NBA_forward_model = linearModel(NBA)
NBA_forward_model.feature_selection('forward')
NBA_forward_model.predict()
print('Forward model R Squared value: ' + str(NBA_forward_model.r_squared))

NBA_backward_model = linearModel(NBA)
NBA_backward_model.feature_selection('backward')
NBA_backward_model.predict()
print('Backward model R Squared value: ' + str(NBA_backward_model.r_squared))

# Pick the best model based on the R squared value
NBA_models = {NBA_full_model: NBA_full_model.r_squared, NBA_forward_model: NBA_forward_model.r_squared,
              NBA_backward_model: NBA_backward_model.r_squared}
NBA_best = max(NBA_models, key=NBA_models.get)
NBA_best_name = [x for x in globals() if globals()[x] is NBA_best][0]
print('Based on the R Squared metric, the ' + str(NBA_best_name) + ' is the best choice.')
# Print out the statistical summary for the best model
NBA_best.X_train = add_constant(NBA_best.X_train)
NBA_new_model = OLS(NBA_best.Y_train, NBA_best.X_train).fit()
print(NBA_new_model.summary())
print("==============")

# for league in df:
name = [x for x in globals() if globals()[x] is WNBA][0]
print("======" + str(name) + "======")
WNBA_full_model = linearModel(WNBA)
WNBA_full_model.predict()
print('Full model R Squared Valued: ' + str(WNBA_full_model.r_squared))

WNBA_forward_model = linearModel(WNBA)
WNBA_forward_model.feature_selection('forward')
WNBA_forward_model.predict()
print('Forward model R Squared value: ' + str(WNBA_forward_model.r_squared))

WNBA_backward_model = linearModel(WNBA)
WNBA_backward_model.feature_selection('backward')
WNBA_backward_model.predict()
print('Backward model R Squared value: ' + str(WNBA_backward_model.r_squared))

# Pick the best model based on the R squared value
WNBA_models = {WNBA_full_model: WNBA_full_model.r_squared, WNBA_forward_model: WNBA_forward_model.r_squared,
               WNBA_backward_model: WNBA_backward_model.r_squared}
WNBA_best = max(WNBA_models, key=WNBA_models.get)
WNBA_best_name = [x for x in globals() if globals()[x] is WNBA_best][0]
print('Based on the R Squared metric, the ' + str(WNBA_best_name) + ' is the best choice.')
# Print out the statistical summary for the best model
WNBA_best.X_train = add_constant(WNBA_best.X_train)
WNBA_new_model = OLS(WNBA_best.Y_train, WNBA_best.X_train).fit()
print(WNBA_new_model.summary())
# Plot the correlation matrix between all the predictors
print("==============")
