from Transformar import Datasets, load_creado

Datas = Datasets()
arch = open("tabla.csv", "w")
arch.write("Dataset,Instancias,Caracteristicas,Positivos,Negativos\n")
for data in Datas:
    print(data.name)
    X,y = load_creado(data)
    print(y)
    bandera = True if 2 in y else False
    contador0 = 0
    contador1 = 0
    if bandera == True:
        for i in y:
            if i == 1:
                contador0 = contador0+1
            else:
                contador1 = contador1+1
    else:
        for i in y:
            if i == 0:
                contador0 = contador0+1
            else:
                contador1 = contador1+1 
              
    arch.write(f"{data.name},{len(X)},{len(X[0])},{contador1},{contador0}\n")
    
arch.close()
                
        
    