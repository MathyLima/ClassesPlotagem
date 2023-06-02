import mathyPlot as mtp
import numpy as np
primeiras_coord = mtp.Coordenadas(60,30,36,144,2)
segunda_coord = mtp.Coordenadas(20,40,16,16,2)
terceira_coord = mtp.Coordenadas(25,10,100,25,2)


primeira_classe=mtp.Classes()
primeira_classe.coordenadas= primeiras_coord.coordenadas
segunda_classe = mtp.Classes()
print(primeira_classe.coordenadas)
print(primeira_classe.get_y_Class)
segunda_classe.coordenadas = segunda_coord.coordenadas
terceira_classe = mtp.Classes()
terceira_classe.coordenadas = terceira_coord.coordenadas
teste = mtp.Plota()
teste.classes = primeira_classe,segunda_classe,terceira_classe

teste.plota()

