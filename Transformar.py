import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score, recall_score, f1_score,precision_score
from prueba2 import media_desvia
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
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
#sirve para partir los datos entre datos de prueba, y de entrenamiento
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

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

def metrics(y_true, y_pred, Lista):
    Lista[0].append(accuracy_score(y_true, y_pred))
    Lista[1].append(recall_score(y_true,y_pred, zero_division=0,average="macro"))
    Lista[2].append(f1_score(y_true,y_pred, zero_division=0, average="macro"))
    Lista[3].append(precision_score(y_true, y_pred, zero_division=0, average="macro"))

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
    ('3nn',KNeighborsClassifier(n_neighbors=3)),
    ('gnb', GaussianNB())]
    
kf = KFold(n_splits=10)
#kf = StratifiedShuffleSplit(n_splits=10, test_size=0.9, random_state=1)
for Dataset in Datasets():
    X,y = load_creado(Dataset)
    Lista_het = [[],[],[],[]]
    Lista_homo = [[],[],[],[]]
    Lista_ada = [[],[],[],[]]
    Lista_rand = [[],[],[],[]]
    for train_index, test_index in kf.split(X,y):
        het = Heterogeneo(estimadores).fit(X[train_index],y[train_index])
        y_pred_het = het.predict(X[test_index])
        bag = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3), n_estimators=100).fit(X[train_index],y[train_index])
        y_pred_bag = bag.predict(X[test_index])
        ada = AdaBoostClassifier(n_estimators=100, random_state=True).fit(X[train_index],y[train_index])
        y_pred_ada = ada.predict(X[test_index])
        rand = RandomForestClassifier(n_estimators=100 , max_depth=5, bootstrap=True).fit(X,y)
        y_pred_rand = rand.predict(X[test_index])
        
        metrics(y[test_index], y_pred_het, Lista_het)
        metrics(y[test_index], y_pred_bag, Lista_homo)
        metrics(y[test_index], y_pred_ada, Lista_ada)
        metrics(y[test_index], y_pred_rand, Lista_rand)
    heter = media_desvia(Lista_het)
    homo = media_desvia(Lista_homo)
    adab = media_desvia(Lista_ada)
    rando = media_desvia(Lista_rand)
    print("--- Dataset: ", Dataset.name)
    print("metricas heterogeneo: medias:", heter.media()," desviaciones estandar: ", heter.desviacion()) 
    print("metricas homogeneo: medias:", homo.media()," desviaciones estandar: ", homo.desviacion()) 
    print("metricas adaboost: medias:", adab.media()," desviaciones estandar: ", adab.desviacion()) 
    print("metricas random forrest: medias:", rando.media()," desviaciones estandar: ", rando.desviacion()) 

