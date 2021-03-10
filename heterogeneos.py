class Homogeneo:
    def __init__(self, estimadores):
        self.estimadores = estimadores

    def fit(X,y):
        for modelo, estimador in self.estimadores:
        estimador.fit(X,y)
    
    def predict(X):
        n_estimadores = len(estimadores)
        n_samples = len(X)
    
        y = np.zeros((n_samples, n_estimadores))
        for i,(modelo,estimador) in enumerate(estimadores):
            y[:, i] = estimador.predict(X)
            y = mode(y, axis=1)
            y = [i[0] for i in y[0]]
    return y