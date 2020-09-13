import numpy as np
from collections import Counter
class KNN:
    def __init__(self, neighbors=5):
        self.neighbors = neighbors      
        
    def fit(self, x_train, y_train):
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        
    def distances(self, X):
        x = np.array(X)
        distancelist = []
        for index, data_point in enumerate(self.x_train):
            data_point = np.array(data_point)
            distance = np.sqrt(np.sum((data_point - x)**2))
            distancelist.append([distance, self.y_train[index]])
            
        return distancelist
            
    def classs(self, distancelist):
        distances_sorted = sorted(distancelist)[:self.neighbors]
        label = Counter([distance[1] for distance in distances_sorted]).most_common(1)[0][0]
        
        return [label]
    
    def predict(self, X):
        predicted = []
        for i in X:
            distances = self.distances(i)
            predicted.append(self.classs(distances))
        return np.array(predicted)
    
    def score(self, X, y):
        predicted = self.predict(X)
        y = y.reshape(-1,1)
        correct= predicted==y
        
        return np.mean(correct)