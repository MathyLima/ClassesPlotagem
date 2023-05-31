import matplotlib.pyplot as plt
import numpy as np

class Coordenadas:
    _linha = 0
    @classmethod
    def get_total_linhas(cls):
        return cls._linha
 
    def __init__(self,x=0,y=0,z=0):
        self._x = x
        self._y = y
        self._z = z
        Coordenadas._linha += 1
        self.linha = Coordenadas._linha
    
    @property
    def coordenada_ponto(self):
        return [self.linha,self._x,self._y,self._z]
    
    def seta_pontos(self,x,y,z=0):
        self._x=x
        self._y=y
        self._z=z

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
            self.coordenadas.append(coordenadas.coordenada_ponto)
        
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
        
    #retorna uma tupla contendo as classes criadas, essas que por sua vez armzenam seus proprios pontos
    @property
    def classes(self):
        return np.array(self._classes)
    @classes.setter
    def classes(self,classe):
        for classes in classe:    
            self._classes.append(classes.coordenadas)
            
    
    @property
    def get_X(self):        
        return np.array(list((ponto[1])for classes in self.classes for ponto in classes))   
    @property
    def get_Y(self):
        return np.array(list((ponto[2])for classes in self.classes for ponto in classes))
        
  
    def plota(self):
        #  essa funcao deve percorrer o array de classes, que receberar todas as classes, que terao armazenados os pontos, a partir disso os pontos sera distribuidos e plotados
        #  sendo assim deve haver uma junção de graficos, cada um contendo os pontos de cada classe
        classe_1=self.classes[0]
        print(classe_1)
        cor_pontos=['r','g','b']
        fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
        ax.set_xlabel('Posicao X')
        ax.set_ylabel('Posicao Y')
        for i in range(self.totalClasses):
            data_pontos={
                'x':np.array(list((ponto[1])for ponto in self.classes[i])),
                'y':np.array(list((ponto[2])for ponto in self.classes[i]))
            }
            data={
                'x':data_pontos['x'],
                'y':data_pontos['y']
                }
            print(data['x'])
            cor_ponto=list(cor_pontos[i])
            ax.scatter('x','y',color=cor_ponto,data=data)

        
        plt.show()