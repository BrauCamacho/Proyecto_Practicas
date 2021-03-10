import numpy as np
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
class Heterogeneo(BaseEstimator, ClassifierMixin):
    def __init__(self, estimadores):
        self.estimadores = estimadores

    def fit(self,X,y):
        X, y = check_X_y(X, y)
        
        self.classes_ = unique_labels(y)
        for modelo, estimador in self.estimadores:
            estimador.fit(X,y)
        return self
    
    def predict(self,X):
        check_is_fitted(self)
        X = check_array(X)
        n_estimadores = len(self.estimadores)
        n_samples = len(X)
        y = np.zeros((n_samples, n_estimadores))
        for i,(modelo,estimador) in enumerate(self.estimadores):
            y[:, i] = estimador.predict(X)
        y = mode(y, axis=1)
        y = [i[0] for i in y[0]]
        return y