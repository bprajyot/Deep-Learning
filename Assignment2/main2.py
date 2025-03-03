import pandas as pd
import numpy as np
import operator
import warnings
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

df = pd.read_csv('Assignmet 2/Iris.csv')

def euclidian(arr1, arr2, col):
    distance = 0
    for i in range(col):
        distance += np.square(arr1[i] - arr2[i])
    return np.sqrt(distance)

def knn(train, test, k):
    distance = {}
    col = test.shape[1]  

    for i in range(len(train)):
        dis = euclidian(train.iloc[i], test.iloc[0], col-1)  
        distance[i] = dis

    SortedDistance = sorted(distance.items(), key=operator.itemgetter(1))

    neighbour = []
    for i in range(k):
        neighbour.append(SortedDistance[i][0])
        
    classvotes = {}
    for i in range(len(neighbour)):
        ClassifiedAs = train.iloc[neighbour[i], -1]  
        if ClassifiedAs in classvotes:
            classvotes[ClassifiedAs] += 1
        else:
            classvotes[ClassifiedAs] = 1
    
    SortedVotes = sorted(classvotes.items(), key=operator.itemgetter(1), reverse=True)
    
    return SortedVotes[0][0], neighbour

testing = pd.DataFrame([[7.2, 3.6, 5.1, 2.5]])

kvalue = 5

result, nb = knn(df, testing, kvalue)

print(f"Predicted Class: {result}, Neighbors: {nb}")

nbb = KNeighborsClassifier(n_neighbors = 5 )
nbb.fit(df.iloc[:,0:4], df['Species'])
print(nbb.predict(testing))
print(nbb.kneighbors(testing)[1][0])