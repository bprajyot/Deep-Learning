import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
warnings.filterwarnings('ignore')

df = pd.read_csv('car_evaluation.csv')

df.columns = ['Purchase', 'Demand', 'Number of Doors', 'Seating Capacity', 'Luggage Capacity', 'Safety', 'Class']

# Checking the dataset
# print(df.head())
# print(df['Class'].value_counts())
# print(df.isnull().sum())

x = df.drop(['Class'], axis=1)
y = df['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
# print(x_train.head())

encoder = ce.OrdinalEncoder(cols=['Purchase', 'Demand', 'Number of Doors', 'Seating Capacity', 'Luggage Capacity', 'Safety'])

x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)

rf = RandomForestClassifier(random_state=0) 
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
print('{0:0.4f}'. format(accuracy_score(y_test, y_pred)))

rf2 = RandomForestClassifier(n_estimators=100, random_state=0)
rf2.fit(x_train, y_train)
y_pred2 = rf2.predict(x_test)
print('{0:0.4f}'. format(accuracy_score(y_test, y_pred2)))

feature_scores = pd.Series(rf.feature_importances_, index=x_train.columns).sort_values(ascending=False)
print(feature_scores)
# can drop the features with least feature score to optimize the model even further


cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)

print(classification_report(y_test, y_pred))