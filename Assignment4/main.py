import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

df = pd.read_csv('Iris.csv')

d = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df['Species'] = df['Species'].map(d)

print(df.head())

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

x = df[features]
y = df['Species']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(x, y)

plot_tree(dtree, class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], filled=True)

plt.show()