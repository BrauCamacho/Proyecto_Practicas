import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import precision_score


def adaboost(X,y, n_estimators): #entrena los estimadores, usando la metodología adaboost
    n_samples, n_features = X.shape
    pesos = np.ones((n_samples,)) #inicializa los pesos en 1
    estimadores =[] 
    for t in range(n_estimators):
        pesos = pesos/np.sum(pesos) #normaliza todo
        h = DecisionTreeClassifier(max_depth=1) #inicializa el clasificador
        h.fit(X,y, sample_weight=pesos) #lo entrena
        ypred = h.predict(X) #genera un modelo basado en las predicciones de un alumno debil
        e = 1 - precision_score(y_true =y,y_pred= ypred, sample_weight= pesos)
        a = 0.5 * np.log((1-e)/e)
        m = (m == ypred)*1 +(y != ypred)*-1
        pesos*= np.exp(-a*m)
        estimadores.append((a,h))
    return estimadores

def predict_boosting(X, estimadores):
    pred = np.zeros((len(X),))
    for a, h in estimadores:
        pred+= a * h.predict(X)
    y = np.sign(pred)
    return y
    
X,y = load_iris(return_X_y=True)
estimadores = adaboost(X,y, 10)
y = predict_boosting(X, estimadores)
print(y)