import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as regression

dataset = pd.read_csv('Dataset.csv')
# print(dataset.head())

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[: ,1].values
# print(X)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

regressor = regression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
# print(y_pred)

#plot for the TEST
 
plt.scatter(x_test, y_test, color='red') 
plt.plot(x_train, regressor.predict(x_train), color='blue') # plotting the regression line
 
plt.title("Salary vs Experience (Testing set)")
 
plt.xlabel("Years of experience") 
plt.ylabel("Salaries") 
plt.show() 