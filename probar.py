from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score,precision_score, classification_report

X,y = load_iris(return_X_y=True)
kn = KNeighborsClassifier(n_neighbors=3).fit(X,y)
y_pred = kn.predict(X)
res = recall_score(y, y_pred,average=None)
print(res)

