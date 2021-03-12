import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
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
    def __init__(self, dir, clas,name):
        self.dir = dir
        self.clas = clas
        self.name = name

    
#crea los ensambles homogeneos
def ensambles_homogeneos(Estimador_Base, cantidad):
    estimadores = [Estimador_Base for i in range(cantidad)]
    return estimadores

#permite cargar cualquier dataset, con la estructura de sklearn
def load_creado(datos):
    dat = open(datos.dir)
    X = np.array([np.fromstring(i, dtype = float, sep = ',') for i in dat.readlines()])
    y = np.array([i[datos.clas]for i in X])
    X = np.array([[vector[i] for i in range(len(vector)) if i != datos.clas] for vector in X])
    return X,y

#todos los datasets, con la direccion de la clase
def Datasets():
    datasets =[]
    datasets.append(datas("Datasets/bupa.data",6,"Liver disorders"))
    datasets.append(datas("Datasets/diabetic_retinopathy_debrecen.txt", 0,"Diabetic retinopathy from U. of Debrecen"))
    datasets.append(datas("Datasets/fertility_Diagnosis2.txt", 9,"Fertility"))
    datasets.append(datas("Datasets/haberman.data", 3,"Haberman’s survival after surgery for breast cancer"))
    datasets.append(datas("Datasets/heart.txt", 13,"Statlog heart disease"))
    datasets.append(datas("Datasets/LSTV.txt", 310,"LSVT voice rehabilitation"))
    datasets.append(datas("Datasets/ThoraricSurgery.txt",16,"Thoracic surgery survival after surgery for lung cancer"))
    datasets.append(datas("Datasets/vertebral_column.txt",6,"Vertebral column"))
    datasets.append(datas("Datasets/train_data.txt",28,"Parkinson speech"))
    datasets.append(datas("Datasets/wdbc.txt",1,"Breast cancer Wisconsin diagnostic"))
    datasets.append(datas("Datasets/wdbcOriginal.txt",1,"Breast cancer Wisconsin original"))
    datasets.append(datas("Datasets/wdbcPredictions.txt",1,"Breast cancer Wisconsin prognostic"))
    datasets.append(datas("Datasets/chronic_kidney_disease.txt",24,"Chronic kidney disease"))
    datasets.append(datas("Datasets/parkinsons.data",0, "Oxford Parkinson’s disease detection"))
    return datasets

#Carga todos los estimadores, en una lista
estimadores = [
    ('dt',DecisionTreeClassifier(max_depth=5)),
    ('svm', SVC(gamma=1.0, C=1.0, probability=True)),
    #('gp', GaussianProcessClassifier(RBF(1.0))),
    ('3nn',KNeighborsClassifier(n_neighbors=3)),
    ('gnb', GaussianNB())
]
kf = KFold(n_splits=10)
for Dataset in Datasets():
    X,y = load_creado(Dataset)
    media_het = 0
    media_ada = 0
    media_bag = 0
    for train_index, test_index in kf.split(X):
        het = Heterogeneo(estimadores).fit(X[train_index],y[train_index])
        y_pred_het = het.predict(X[test_index])
        bag = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3), n_estimators=100).fit(X[train_index],y[train_index])
        y_pred_bag = bag.predict(X[test_index])
        ada = AdaBoostClassifier(n_estimators=100, random_state=True).fit(X[train_index],y[train_index])
        y_pred_ada = ada.predict(X[test_index])
        media_het+= accuracy_score(y[test_index], y_pred_het)
        media_ada+= accuracy_score(y[test_index], y_pred_ada)
        media_bag+= accuracy_score(y[test_index], y_pred_bag)
    media_het = media_het/10
    media_ada = media_ada/10
    media_bag = media_bag/10
    print("---- Resultados de presicion para el dataset : ",Dataset.name," --------" )
    print("Estimador : Bagging, Presicion: ",media_bag)
    print("Estimador : Adaboost, Presicion: ",media_ada)
    print("Estimador : Heterogeneo, Presicion: ",media_het)
