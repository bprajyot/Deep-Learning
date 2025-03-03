import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as regression

dataset = pd.read_csv('Dataset.csv')
df = dataset[['area', 'price']]
df100 = df[:][:600]
  
x = np.array(df100['area']).reshape(-1, 1) 
y = np.array(df100['price']).reshape(-1, 1) 
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

regressor = regression()
regressor.fit(x_train, y_train)
print(regressor.score(x_test, y_test))

y_pred = regressor.predict(x_test) 
plt.scatter(x_test, y_test, color ='blue') 
plt.plot(x_test, y_pred, color ='red') 
plt.title("Area v/s Price")
plt.xlabel("Area")
plt.ylabel("Price")
plt.grid(True)

plt.show() 