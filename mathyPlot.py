import matplotlib.pyplot as plt
import numpy as np

class Coordenadas:
    _linha = 0
    @classmethod
    def get_total_linhas(cls):
        return cls._linha
 
    def __init__(self,x,y,z=0):
        self._x = x
        self._y = y
        self._z = z
        Coordenadas._linha += 1
        self.linha = Coordenadas._linha
    
    @property
    def coordenada_ponto(self):
        return [self.linha,self._x,self._y,self._z]

#gera elementos que sao taxados como sendo de mesma classe
class Classes:
    _classes = 0
    
    @classmethod
    def get_total_classes(cls):
        return cls._classes
    
    def __init__(self,*args): 
        self._coordenadas=[]
        self._quantidade_linhas = Coordenadas.get_total_linhas()
        Classes._classes += 1
        self.classe = Classes._classes
    
    @property    
    def coordenadas(self):
        return self._coordenadas
    @coordenadas.setter
    def coordenadas(self,coordenada):
        for coordenadas in coordenada:
            self.coordenadas.append(coordenadas)
        
    @property
    def get_x_Class(self):
        return np.array([(coordenadaX[1]) for coordenadas in self.coordenadas for coordenadaX in coordenadas])

    @property
    def get_y_Class(self):
        return np.array([(coordenadaY[2]) for coordenadas in self.coordenadas for coordenadaY in coordenadas ])
    


class Plota():
    def __init__(self):
        self._classes = []
        self.totalClasses = Classes.get_total_classes()
        
    @property
    #retorna uma tupla contendo as classes criadas, essas que por sua vez armzenam seus proprios pontos
    def classes(self):
        return np.array(self._classes)
    @classes.setter
    def classes(self,classe):
        for classes in classe:    
            self._classes.append(classes)
            
    
    @property
    def get_X(self):        
        return np.array(list((ponto[1])for classes in self.classes for ponto in classes))   
    @property
    def get_Y(self):
        return np.array(list((ponto[2])for classes in self.classes for ponto in classes))
        
  
    def plota(self):
        terceira_classe=False
        
        #  essa funcao deve percorrer o array de classes, que receberar todas as classes, que terao armazenados os pontos, a partir disso os pontos sera distribuidos e plotados
        #  sendo assim deve haver uma junção de graficos, cada um contendo os pontos de cada classe
        classe_1=self.classes[0]
        print(classe_1)
        
        print(self.get_X)
        pegaPontosClasse_1= {'x': np.array(list((ponto[1])for ponto in classe_1)),
                             'y': np.array(list((ponto[2])for ponto in classe_1))
                             }   
        
        data_primeira_classe={'x': pegaPontosClasse_1['x'],
                            'y': pegaPontosClasse_1['y']}
        
        classe_2=self.classes[1]
        pegaPontos_Classe_2={'x':np.array(list((ponto[1])for ponto in classe_2)),
                             'y':np.array(list((ponto[2])for ponto in classe_2))}
        data_segunda_classe={'x':pegaPontos_Classe_2['x'],
                            'y':pegaPontos_Classe_2['y']}
        print(pegaPontosClasse_1['x'])
        
        # pegaClasse_2 = np.array((classe[1]) for classe in self.classes)
        # pegaPontosClasse_2 = {'x': np.array((coordenadas[1]) for coordenadas in self.classes),
        #                       'y': np.array((coordenadas[2]) for coordenadas in self.classes)}
        
        if len(self.classes) >2:
             terceira_classe=True
             classe_3 = self.classes[2]
             pegaPontos_Classe_3 = {'x': np.array(list((ponto[1])for ponto in classe_3)),
                                    'y': np.array(list((ponto[2]) for ponto in classe_3))}
             data_terceira_classe = {
                                    'x':pegaPontos_Classe_3['x'],
                                    'y':pegaPontos_Classe_3['y']}
            
        
        
        
        #print(data_primeira_classe['x'])
        fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
        ax.scatter('x','y',color='g',data=data_primeira_classe)
        ax.scatter('x','y',color='r',data=data_segunda_classe)
        if terceira_classe==True:
            ax.scatter('x','y',color='b',data=data_terceira_classe)
        ax.set_xlabel('Posicao X')
        ax.set_ylabel('Posicao Y')
    
        plt.show()