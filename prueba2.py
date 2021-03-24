import statistics as stats
import numpy as np
class media_desvia:
    def __init__(self, Lista):
        self.lista = np.array(Lista)
    def media(self):
       return [np.mean(self.lista[:,i]) for i in range(1, len(self.lista[0]))]
    def desviacion(self):
        return [np.std(self.lista[:,i]) for i in range(1,len(self.lista[0]))]