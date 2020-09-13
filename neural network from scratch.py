import numpy as np
class neuralnetworks:
  def __init__(self,alpha=1,iterations=2000,n_hiddennodes=100):
    self.alpha=alpha
    self.iterations=iterations
    self.n_hiddennodes=n_hiddennodes
    self.w1=None
    self.w2=None
    self.b1=None
    self.b2=None
    self.predictions=None
  def activate(self,z):
    return 1/(1+np.exp(-z)) 
  def forwardprop(self,x_train,y_train):
    n_output,n_samples=y_train.shape
    n_features=x_train.shape[0]
    self.w1=np.random.randn(self.n_hiddennodes,n_features)
    self.b1=np.zeros((self.n_hiddennodes,1))
    self.w2=np.random.randn(n_output,self.n_hiddennodes)
    self.b2=np.zeros((n_output,1))
  def loss(self,y_train,y_predicted):
    loss=-np.sum(np.multiply(y_train,np.log(y_predicted)))/y_train.shape[1]
    return loss
  def backprop(self,y_train,x_train):
    for i in range(2000):
      Z1 = np.matmul(self.w1,x_train) + self.b1
      A1 = self.activate(Z1)
      Z2 = np.matmul(self.w2,A1) + self.b2
      A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)
      self.predictions=A2
      cost = self.loss(y_train, A2)
      m=y_train.shape[1]
      dZ2 = A2-y_train
      dW2 = (1/m) * np.matmul(dZ2, A1.T)
      db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

      dA1 = np.matmul(self.w2.T, dZ2)
      dZ1 = dA1 * self.activate(Z1) * (1 - self.activate(Z1))
      dW1 = (1/m) * np.matmul(dZ1, x_train.T)
      db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

      self.w2 -=self.alpha * dW2
      self.b2 -=self.alpha * db2
      self.w1 -=self.alpha * dW1
      self.b1 -=self.alpha * db1

    print("Final cost:", cost)
   
  def train(self,x_train,y_train):
    self.forwardprop(x_train,y_train)
    self.backprop(y_train,x_train)
    self.accuracy(y_train)
  def accuracy(self,y_train):
    from sklearn.metrics import classification_report
    self.prediction = np.argmax(self.predictions, axis=0)
    labels = np.argmax(y_train, axis=0)
    print(classification_report(self.prediction, labels))
    print("acc=",np.mean(self.predictions.round()==y_train)*100)
                                                                          
    

