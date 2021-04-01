import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score, recall_score, f1_score,precision_score, classification_report
from imblearn.metrics import sensitivity_score
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
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier

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

def metrics(y_true, y_pred, estimador, datos):
    rc = recall_score(y_true,y_pred, zero_division=0,average=None)
    #rc = np.append(rc, 0)
    f1 = f1_score(y_true,y_pred, zero_division=0, average=None)
    #f1 = np.append(f1, 0)
    p = precision_score(y_true, y_pred, zero_division=0, average=None)
    #p = np.append(rc, 0)
    return f"{estimador},{datos},{accuracy_score(y_true, y_pred)},{rc[0]},{rc[1]},{f1[0]},{f1[1]},{p[0]},{p[1]}\n"
#todos los datasets, con la direccion de la clase
def Datasets():
    datasets =[]
    datasets.append(datas("Datasets/bupa.data",6,"Liver disorders"))
    datasets.append(datas("Datasets/diabetic_retinopathy_debrecen.txt", 1,"Diabetic retinopathy from U. of Debrecen"))
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
    datasets.append(datas("Datasets/parkinsons.data",17, "Oxford Parkinson’s disease detection"))
    return datasets

#Carga todos los estimadores, en una lista
estimadores = [
    ('dt',DecisionTreeClassifier(max_depth=5)),
    ('svm', SVC(gamma=1.0, C=1.0, probability=True)),
    ('3nn',KNeighborsClassifier(n_neighbors=3)),
    ('gnb', GaussianNB())
    
    ]

metaestimadores = [
    [StackingClassifier(estimators = estimadores, cv = 10), "StackingClassifier"],
    [VotingClassifier(estimadores), "votingClassifier"],
    #[BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3), n_estimators=100), "Bagging"],
    [AdaBoostClassifier(n_estimators=100, random_state=True), "AdaBoost"],
    [RandomForestClassifier(n_estimators=100 , max_depth=5, bootstrap=True), "RandomForest"]
]
    
#kf = KFold(n_splits=10)
kf = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=1)
for Dataset in Datasets():
    X,y = load_creado(Dataset)
    res = open(f"Resultados/{Dataset.name}.csv", "w")
    res.write(f"Estimador,Datos,Accuracy_score,recall_score_1,recall_score_2,f1_score_1,f1_score_2,precision_score_1,precision_score_2\n")
    for meta in metaestimadores:
        for train_index, test_index in kf.split(X,y):
            meta[0].fit(X[train_index],y[train_index])
            train = meta[0].predict(X[train_index])
            res.write(metrics(y[train_index], train, meta[1], "Entrenamiento"))
            test = meta[0].predict(X[test_index])
            res.write(metrics(y[test_index], test, meta[1], "Prueba"))
    res.close()