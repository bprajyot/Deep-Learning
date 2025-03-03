from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

iris = pd.read_csv('Iris.csv')
d = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
iris['Species'] = iris['Species'].map(d)

print(iris.head())

# features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# X = iris[features]
# y = iris['Species']

X = iris.iloc[:,:-1]
y = iris.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

with open('model.pkl', 'wb') as model_file:
    pickle.dump(nb_classifier, model_file)
    print("Model saved as nb_classifier.pkl")

X_test.to_csv('test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
# y_pred.to_csv('y_pred.csv', index=False)