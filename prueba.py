from sklearn.ensemble import AdaBoostClassifier
from Transformar import datas
from Transformar import load_creado
from Transformar import Datasets
from sklearn.metrics import accuracy_score

datos = Datasets()
for datas in datos:
    X,y = load_creado(datas)
    ab = AdaBoostClassifier(n_estimators=100, random_state=True).fit(X,y)
    ypred = ab.predict(X)
    print(accuracy_score(y , ypred))