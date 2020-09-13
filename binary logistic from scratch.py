import numpy as np
class LogisticRegression(object):
    def __init__(self,x,y,alpha=0.01,no_of_iterations=10000):
        self.m = x.shape[0]
        self.n = x.shape[1]
        self.alpha = alpha
        self.x = x
        self.no_of_iterations = no_of_iterations
        self.x= self.x - (self.x).mean(axis=0)
        max = np.abs(self.x).max(axis=0)
        max[max==0] = 1
        self.x = self.x / max
        self.y =y
        self.para = np.random.randn(self.n)
        self.c = 0
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z)) 
    def loss(self,y_pred,y):
        loss =(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)).sum()
        return loss;

    def fit(self):
        self.y_pred = (self.x).dot(self.para) + self.c
        self.y_pred = self.sigmoid(self.y_pred).round()
        for i in range(self.no_of_iterations):
            self.para = self.para -(self.alpha/self.m)*(self.x.T).dot(self.y_pred - self.y)
            self.c-=self.alpha*np.sum(self.y_pred-self.y)/(self.m)
            self.y_pred = (self.x).dot(self.para) + self.c
            self.y_pred = self.sigmoid(self.y_pred).round()
        self.y_pred = (self.x).dot(self.para) + self.c  
        self.y_pred = self.sigmoid(self.y_pred).round()
        loss = self.loss(self.y_pred,self.y)
        return loss
    def pred(self, X=None,Y = None):
        if (X==None):
            X = self.x
        else:
            X= X - (X).mean(axis=0)
            max = np.abs(X.max(axis=0))
            max[max==0] = 1
            X = X / max
        if(Y==None):
            Y = self.y
        y_pred = (X).dot(self.para) + self.c
        y_pred = self.sigmoid(y_pred).round()
        return y_pred

    def para(self):
        return self.para
    def constant(self):
        return self.c
    def score(self,X=None,Y = None):
        y_pred = self.pred(X,Y)
        correct= (y_pred==Y)
        return np.mean(correct)