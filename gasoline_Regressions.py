# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:53:29 2019

@author: Manash
"""

# Simple Linear Regression of Gasoline

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dataset = pd.read_csv('gasoline.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)



# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Quantity vs Years (Training set)')
plt.xlabel('Years')
plt.ylabel('Quanity')
plt.show()


# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Quantity vs Years (Test set)')
plt.xlabel('Years')
plt.ylabel('Quanity')
plt.show()


# optional

print(y_pred)

# print m = ?
print(regressor.coef_)

#print C = ?
print(regressor.intercept_)

#Value of R^2
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)




