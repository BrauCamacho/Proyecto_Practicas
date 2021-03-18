import statistics as stats
class media_desvia:
    def __init__(self, Lista):
        self.lista = Lista
    def media(self):
        return [stats.mean(lis) for lis in self.lista]
    def desviacion(self):
        return [stats.stdev(lis) for lis in self.lista]