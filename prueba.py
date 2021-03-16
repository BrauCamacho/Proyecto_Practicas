from sklearn.preprocessing import StandardScaler
from Transformar import Datasets
from Transformar import load_creado
from sklearn.neural_network import MLPClassifier

Datas = Datasets()
for data in Datas:
    X,y = load_creado(data)
    scaled = StandardScaler().fit(X)
    X_scaled = scaled.transform(X)
    nn = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1, max_iter=10000).fit(X_scaled,y)
