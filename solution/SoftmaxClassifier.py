from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class SoftmaxClassifier(BaseEstimator, ClassifierMixin):  
    """A softmax classifier"""

    def __init__(self, lr = 0.1, alpha = 100, n_epochs = 1000, eps = 1.0e-5,threshold = 1.0e-10 , regularization = True, early_stopping = True):
       

        self.lr = lr
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.eps = eps
        self.regularization = regularization
        self.threshold = threshold
        self.early_stopping = early_stopping
        


    def fit(self, X, y=None):
        
        prev_loss = np.inf
        self.losses_ = []
        
#         self.nb_example = X.shape[0]
        self.nb_feature = X.shape[1]
        self.nb_classes = len(np.unique(y))

        

        X_bias = np.c_[np.ones((X.shape[0])), X ] 
        
        self.theta_  = np.random.normal(scale = 0.3,size=(self.nb_feature+1,self.nb_classes))
        

        for epoch in range( self.n_epochs):

            z = X_bias.dot(self.theta_)
            probas = self._softmax(z)
            
            
            loss = self._cost_function(probas, y )    
            
            self.theta_ = self.theta_ - self._get_gradient(X_bias,y,probas)
            
            self.losses_.append(loss)

            if np.abs(loss - prev_loss) < self.threshold:
                print("stopped at epoch n" + str(epoch))
                break
            else:
                prev_loss = loss


        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X,y)
    
    
    
    def predict(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        
        # X_bias = np.c_[np.ones((X.shape[0])), X ] 
        # z = np.matmul( X_bias, self.theta_) # m * k
        probabilities = self.predict_proba(X)
        prediction = np.argmax(probabilities, axis = 1)
        
        return prediction
    
    def predict_proba(self, X, y=None):
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        X_bias = np.c_[np.ones((X.shape[0])), X ] 
        z = np.matmul( X_bias, self.theta_) # m * k
        prediction = self._softmax(z)
        
        return prediction

    def score(self, X, y=None):
        b = self.regularization 
        self.regularization = False
        prediction = self.predict_proba(X)
        score = self._cost_function(prediction, y )
        self.regularization = b
        
        return score
    
    
    def _cost_function(self,probas, y ): 
        y_ohe = self._one_hot(y)
        probas = np.maximum(self.eps, np.minimum(np.ones(probas.shape) - self.eps, probas))
        r = 0.
        
        l = -np.mean(np.sum(y_ohe * np.log(probas), axis=1))
        
        if self.regularization:
            r = np.sum(np.square(self.theta_[1:]))/2.0

        return l + self.alpha/float(probas.shape[0]) * r
    

    
    
    
    def _one_hot(self,y):
        
        # if not type(value) is numpy.ndarray:
        try:    
            y = y.reshape(-1)
            y_ohe = np.zeros((y.shape[0],self.nb_classes))

            for i,l in enumerate(y):
    #             print(i,l)
                y_ohe[i,l] = 1.
        except :
            raise TypeError("You must give a numpy array!")
            

        return y_ohe
    
    def _softmax(self,z):
        z = np.subtract(z.T, np.max(z, axis = 1)).T
        return np.exp(z) / np.sum(np.exp(z), axis = 1,keepdims = True)
    

    
    def _get_gradient(self,X,y, probas):
        
        y_ohe = self._one_hot(y)
    
        regularization_term = np.zeros(self.theta_.shape)
        error = (probas - y_ohe)
        gradient =  np.matmul( X.T, error ) / float(X.shape[0])
        
        if self.regularization:
            regularization_term = np.r_[np.zeros([1, self.nb_classes]),self.theta_[1:]]  / float(X.shape[0])
            
        return self.lr * gradient + self.alpha * regularization_term
    
    