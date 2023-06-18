import numpy as np
import matplotlib.pyplot as plt

#gera elementos que sao taxados como sendo de mesma classe
class Rotulos:
    _rotulo = 0
    
    @classmethod
    def get_total_classes(cls):
        return cls._rotulo
    
    def __init__(self,media_x,media_y,variancia_x,variancia_y,pontosGerados,media_z=0,variancia_z=0): 
        
        self._x = media_x
        self._y = media_y
        self._z = media_z
        
        
        self._variancia_x = variancia_x
        self._variancia_y= variancia_y
        self._variancia_z = variancia_z
        
        self._tamanho = pontosGerados
        self.rotulo = Rotulos._rotulo
        Rotulos._rotulo += 1
        
        
        self._coordenadas = any
    
    @property    
    def coordenadas(self):
        x= self.gera_normal_x
        y = self.gera_normal_y
        label = int(self.rotulo)
        matriz_coordenadas = np.zeros((len(x), 2))
        matriz_coordenadas[:, 0] = x
        matriz_coordenadas[:, 1] = y
        
        self._coordenadas = matriz_coordenadas
        return matriz_coordenadas
   
    
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
    def get_tamanho(self):
        return self._tamanho
    
class Conjunto_rotulos:
    def __init__(self,*args):
        self._conjunto = np.array(args)
        self._numRotulos = len(self._conjunto)
        self._matrix = any 
    
    '''Funcão que recebe as matrizes de todos os conjuntos e as transforma em uma só retorna a matriz, não necessita de parâmetros'''
    @property
    def get_matrix(self):
        x = tuple()
        y = tuple()
        for i in self._conjunto:
             x = np.append(x,i.coordenadas[:,0])
             y = np.append(y,i.coordenadas[:,1])
        matriz_coordenadas = np.zeros((len(x), 2))
        
        matriz_coordenadas[:, 0] = x
        matriz_coordenadas[:, 1] = y
        
        self._matrix=matriz_coordenadas
        return matriz_coordenadas
     
        
    '''Retorna a quantidade de elementos que cada rotulo possui'''
    @property
    def contagem_indices(self):
        count_indices_rotulos = []
        for i in self._conjunto:
            count_indices_rotulos.append(i.get_tamanho)
        
        return count_indices_rotulos
            
   
    '''Função para plotagem, pode ou não receber pontos extras e os plotar'''
    def plotagem(self,x=None,y=None):
        #Vetor com as formas de cada rotulo
        shapes = ['p','s','P']
        #Vetor com as cores de cada rotulo
        cores = ['salmon','khaki','skyblue']
        fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
        ax.set_xlabel('Posição X')
        ax.set_ylabel('Posição Y')
         
        matrix_ponto = self.get_matrix
        count = 0
        count_indices_rotulos = self.contagem_indices
        '''Nesse for, são plotados os valores correspondentes de cada rotulo, para isso fazemos uma partição começando de count = 0, da nossa matriz,
           armazenada em matrix_ponto, até a posição (count + contagem de todos os elementos para cada rotulo),
           sendo assim, se meu rotulo 1 possui 1000 pontos, eu faço a partição de 0 até a posição 999, atualizamos o count com 1000.
           Dessa forma, o próximo rótulo vai começar em 1000, até a posição(tamanho_rotulo -1) e assim por diante
          
          '''     
        for i in range(len(self._conjunto)):    
            data={
                'x':matrix_ponto[count:(count_indices_rotulos[i]+count),0],
                'y':matrix_ponto[count:(count_indices_rotulos[i]+count),1]
                }
          
            ax.scatter('x','y',marker=shapes[i],color=cores[i],edgecolors='k',data=data)
            count += count_indices_rotulos[i]
        
        #A partir daqui, serão adicionados os novos pontos
        data={
            'x':x,
            'y':y
        }
        ax.scatter(x,y,marker='D',color='r',edgecolors='k',data=data)
            
        plt.show()
            
        
        