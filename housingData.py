'''
CRIM - per capita crime rate by town
ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS - proportion of non-retail business acres per town.
CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
NOX - nitric oxides concentration (parts per 10 million)
RM - average number of rooms per dwelling
AGE - proportion of owner-occupied units built prior to 1940
DIS - weighted distances to five Boston employment centres
RAD - index of accessibility to radial highways
TAX - full-value property-tax rate per $10,000
PTRATIO - pupil-teacher ratio by town
B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT - % lower status of the population
MEDV - Median value of owner-occupied homes in $1000's
'''

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

lr = LinearRegression()

df = pd.read_csv("C:/Users/admin/Downloads/archive/housing.csv")
# print(df)

X = df.drop("medv", axis=1)
Y = df["medv"]





X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, test_size=0.3)

train = lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)

print("The mean squared error is:")
print(mean_squared_error(Y_pred, Y_test))
