import statistics as stats
class media_desvia:
    def __init__(self, Lista):
        self.lista = Lista
    def media(self):
        stats.mean(self.lista[0])
        
        return 
    def desviacion(self):
        return [stats.stdev(lis) for lis in self.lista]