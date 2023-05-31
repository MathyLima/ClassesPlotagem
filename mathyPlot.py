import matplotlib.pyplot as plt
import numpy as np

class Coordenadas:
    _linha = 0
    @classmethod
    def get_total_linhas(cls):
        return cls._linha
 
    def __init__(self,media_x,media_y,variancia_x,variancia_y,pontosGerados,media_z=0,variancia_z=0):
        self._x = media_x
        self._y = media_y
        self._z = media_z
        self._variancia_x = variancia_x
        self._variancia_y=variancia_y
        self._variancia_z = variancia_z
        self._tamanho = pontosGerados
        Coordenadas._linha += 1
        self.linha = Coordenadas._linha
    
    @property
    def gera_normal_x(self):
        return np.random.normal(self._x,self.raiz_variancia[0],self._tamanho)
    @property
    def gera_normal_y(self):
        return np.random.normal(self._y,self.raiz_variancia[1],self._tamanho)
   
    @property
    def raiz_variancia(self):
        variancia_x = np.sqrt(self._variancia_x)
        variancia_y = np.sqrt(self._variancia_y)
        variancia_z = np.sqrt(self._variancia_z)
        return variancia_x,variancia_y,variancia_z
    
    @property
    def coordenadas(self):
        return self.gera_normal_x,self.gera_normal_y

    

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
        return np.array([(coordenadas[0]) for coordenadas in self.coordenadas])

    @property
    def get_y_Class(self):
        return np.array([(coordenada[1]) for coordenada in self.coordenadas[1]])
    


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
  
    def plota(self):
        #  essa funcao deve percorrer o array de classes, que receberar todas as classes, que terao armazenados os pontos, a partir disso os pontos sera distribuidos e plotados
        #  sendo assim deve haver uma junção de graficos, cada um contendo os pontos de cada classe
        cor_pontos=['r','g','b']
        fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
        ax.set_xlabel('Posicao X')
        ax.set_ylabel('Posicao Y')
        for i in range(self.totalClasses):
            data_pontos={
                #para cada uma das classes, é acessada sua posicao 0 para X e 1 para Y
                'x':np.array(list((ponto) for ponto in self.classes[i][0])),
                'y':np.array(list((ponto)for ponto in self.classes[i][1]))
            }
            print(data_pontos['x'])
            data={
                'x':data_pontos['x'],
                'y':data_pontos['y']
                }
            print(data['x'])
            cor_ponto=list(cor_pontos[i])
            ax.scatter('x','y',color=cor_ponto,data=data)

        
        plt.show()