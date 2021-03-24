from Transformar import Datasets
from Transformar import load_creado
X,y= load_creado(Datasets()[len(Datasets())-1])
print(X[:,1])