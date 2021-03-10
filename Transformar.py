import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score
#se importan todos los estimadores
from heterogeneos import Heterogeneo
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
#sirve para partir los datos entre datos de prueba, y de entrenamiento
from sklearn.model_selection import train_test_split

#esta clase premite almacenar los datasets, con la ubicación de la clase, como una lista
class datas:
    def __init__(self, dir, clas):
        self.dir = dir
        self.clas = clas
    
#entrena todos los estimadores
def fit(estimadores, X,y):
    for modelo, estimador in estimadores:
        estimador.fit(X,y)
    return estimadores

#crea los ensambles homogeneos
def ensambles_homogeneos(Estimador_Base, cantidad):
    estimadores = [Estimador_Base for i in range(cantidad)]
    return estimadores


    
#genera una prediccion individual por cada clasificador
def predict(X, estimadores):
    n_estimadores = len(estimadores)
    n_samples = len(X)
    
    y = np.zeros((n_samples, n_estimadores))
    for i,(modelo,estimador) in enumerate(estimadores):
        y[:, i] = estimador.predict(X)
    y = mode(y, axis=1)
    y = [i[0] for i in y[0]]
    return y

#permite cargar cualquier dataset, con la estructura de sklearn
def load_creado(datos):
    dat = open(datos.dir)
    X = np.array([np.fromstring(i, dtype = float, sep = ',') for i in dat.readlines()])
    y = np.array([i[datos.clas]for i in X])
    X = [[vector[i] for i in range(len(vector)) if i != datos.clas] for vector in X]
    return X,y

#todos los datasets, con la direccion de la clase
def Datasets():
    datasets =[]
    datasets.append(datas("Datasets/bupa.data",6))
    datasets.append(datas("Datasets/diabetic_retinopathy_debrecen.txt", 0))
    datasets.append(datas("Datasets/fertility_Diagnosis2.txt", 9))
    datasets.append(datas("Datasets/haberman.data", 3))
    datasets.append(datas("Datasets/heart.txt", 13))
    datasets.append(datas("Datasets/LSTV", 309))
    datasets.append(datas("Datasets/TroraricSugery.txt",16))
    datasets.append(datas("Datasets/vertebral_column.txt",6))
    datasets.append(datas("Datasets/train_data.txt",28))
    datasets.append(datas("Datasets/wdbc.txt",1))
    datasets.append(datas("Datasets/wdbcOriginal.txt",1))
    datasets.append(datas("Datasets/wdbcPredictions.txt",1))
    datasets.append(datas("Datasets/chronic_kidney_disease.txt",24))
    datasets.append(datas("Datasets/parkinsons.data",0))
    return datasets

#Carga todos los estimadores, en una lista
estimadores = [
    ('dt',DecisionTreeClassifier(max_depth=5)),
    ('svm', SVC(gamma=1.0, C=1.0, probability=True)),
    ('gp', GaussianProcessClassifier(RBF(1.0))),
    ('3nn',KNeighborsClassifier(n_neighbors=3)),
    ('gnb', GaussianNB())
]
X,y = load_creado(Datasets()[0])
het = Heterogeneo(estimadores)
het.fit(X,y)
y_pred_het = het.predict(X)
bag = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3), n_estimators=100).fit(X,y)
y_pred_bag = bag.predict(X)
ada = AdaBoostClassifier(n_estimators=100, random_state=True).fit(X,y)
y_pred_ada = ada.predict(X)
print("Heterogeneo: ",accuracy_score(y,y_pred_het))
print("Homogeneo: ",accuracy_score(y, y_pred_bag))
print("Boost: ",accuracy_score(y, y_pred_ada))