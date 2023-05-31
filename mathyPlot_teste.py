import mathyPlot as mtp

primeira_coord = mtp.Coordenadas(10,30)
segunda_coord = mtp.Coordenadas(40,50)
terceira_coord = mtp.Coordenadas(60,80)

quarta_coord = mtp.Coordenadas(40,80)
quinta_coord = mtp.Coordenadas(100,40)
sexta_coord = mtp.Coordenadas(80,50)

setima_coord = mtp.Coordenadas(200,90)
oitava_coord = mtp.Coordenadas(74,29)
nona_coord = mtp.Coordenadas(150,4)
primeira_classe=mtp.Classes()

primeira_classe.coordenadas= primeira_coord,segunda_coord,terceira_coord
#print(primeira_classe.coordenadas)
segunda_classe = mtp.Classes()
segunda_classe.coordenadas= quarta_coord,quinta_coord,sexta_coord

terceira_classe = mtp.Classes()
terceira_classe.coordenadas = setima_coord,oitava_coord,nona_coord

teste = mtp.Plota()

teste.classes= primeira_classe,segunda_classe,terceira_classe
#print(type(teste.get_X))

print(teste.classes)

teste.plota()
#print(teste.classes)

