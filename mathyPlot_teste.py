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

primeira_classe.coordenadas= primeira_coord.coordenada_ponto,segunda_coord.coordenada_ponto,terceira_coord.coordenada_ponto
#print(primeira_classe.coordenadas)
segunda_classe = mtp.Classes()
segunda_classe.coordenadas= quarta_coord.coordenada_ponto,quinta_coord.coordenada_ponto,sexta_coord.coordenada_ponto

terceira_classe = mtp.Classes()
terceira_classe.coordenadas = setima_coord.coordenada_ponto,oitava_coord.coordenada_ponto,nona_coord.coordenada_ponto
#print(primeira_classe.get_x)
#print(segunda_classe.get_x)
# print(primeira_coord.coordenada_ponto)
# print(terceira_coord.coordenada_ponto)
# print(primeira_classe.coordenadas)

teste = mtp.Plota()

teste.classes= primeira_classe.coordenadas,segunda_classe.coordenadas,terceira_classe.coordenadas
#print(type(teste.get_X))

print(teste.classes)

teste.plota()
#print(teste.classes)

