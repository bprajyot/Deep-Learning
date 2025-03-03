import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

with open('model.pkl', 'rb') as model:
    load_model = pickle.load(model)

df_test = pd.read_csv('test.csv')
df_test_y = pd.read_csv('y_test.csv')

y_pred_BYPKL = load_model.predict(df_test)

accuracy = accuracy_score(df_test_y, y_pred_BYPKL)
print(f"Accuracy: {accuracy * 100:.2f}%")

d = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
op = y_pred_BYPKL
op = pd.Series(y_pred_BYPKL).map(d)

print(op)