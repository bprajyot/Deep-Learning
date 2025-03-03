import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('Assignmet 2/Iris.csv', sep=',')
# print(df.head(10))

# print(df.isnull().any().values)

# print(df.columns)
df = df.drop("Id", axis= 1)
# print(df.columns)

x = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

n_value = 5

knn = KNeighborsClassifier(n_neighbors=n_value)
knn_model = knn.fit(x_train, y_train)

y_pred = knn_model.predict(x_test)
# print(y_test, y_pred)
print(accuracy_score(y_test, y_pred))

k_params = {"n_neighbors": np.arange(1,50)}

knn = KNeighborsClassifier()

knn_2 = GridSearchCV(knn, k_params, cv=20)
knn_2.fit(x_train, y_train)

print("The best score: " + str(knn_2.best_score_))
print("The best parameters: " + str(knn_2.best_params_))