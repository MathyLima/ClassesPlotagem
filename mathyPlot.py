import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

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
        z = self.gera_normal_z
        label = int(self.rotulo)
        matriz_coordenadas = np.zeros((len(x), 4))
        matriz_coordenadas[:, 0] = x
        matriz_coordenadas[:, 1] = y
        matriz_coordenadas[:, 2] = z
        matriz_coordenadas[:,3] = label
        
        self._coordenadas = matriz_coordenadas
        return matriz_coordenadas
   
    
    @property
    def gera_normal_x(self):
        return np.random.normal(self._x,self.raiz_variancia[0],self._tamanho)
    @property
    def gera_normal_y(self):
        return np.random.normal(self._y,self.raiz_variancia[1],self._tamanho)
    @property
    def gera_normal_z(self):
        return np.random.normal(self._z,self.raiz_variancia[2],self._tamanho)
    @property
    def raiz_variancia(self):
        variancia_x = np.sqrt(self._variancia_x)
        variancia_y = np.sqrt(self._variancia_y)
        variancia_z = np.sqrt(self._variancia_z)
        return variancia_x,variancia_y,variancia_z
    @property
    def get_tamanho(self):
        return self._tamanho
class Join_to_matrix:
    def __init__(self, *args):
        self._conjunto = np.array(args)
        self._numRotulos = len(self._conjunto)
        self._matrix = self.set_matrix
        
    @property
    def set_matrix(self):
        x = tuple()
        y = tuple()
        z = tuple()
        label = tuple()
        for i in self._conjunto:
             x = np.append(x,i.coordenadas[:,0])
             y = np.append(y,i.coordenadas[:,1])
             z = np.append(z,i.coordenadas[:,2])
             label = np.append(label,i.coordenadas[:,3])
        matriz_coordenadas = np.zeros((len(x), 4))
        
        matriz_coordenadas[:, 0] = x
        matriz_coordenadas[:, 1] = y
        matriz_coordenadas[:,2] = z
        matriz_coordenadas[:, 3] = label
        return matriz_coordenadas
     
    @property
    def get_matrix(self):
        return self._matrix
    
class Plot_data:
    def __init__(self,master,data,target, epsilon = None,alpha = None, dimension = None):
        self.data = data
        self.target = target
        self.matrix = self.get_matrix
        self.epsilon = epsilon
        self.alpha = alpha
        self.dimension = dimension
        self.master = master
        self.matrix_reduction = None
        self.value_x = None #valor aleatório(x) que será gerado
        self.value_y = None #agora em y
        self.value_z = None 
        self.value = [self.value_x,self.value_y]
        self.generate_button_radom = tk.Button(
            self.master, text="Gerar ponto aleatório", command=self.make_random_point
        )
        self.generate_button_fill_matrix = tk.Button(
            self.master, text="Preencha a matriz", command=self.fill_the_matrix
        )
        self.generate_button_random_dm = tk.Button(
            self.master, text = "Ponto Aleatório matriz reduzida" , command=self.random_reduction_points
        )
        self.generate_button_classifier = tk.Button(
            self.master, text="Classifique KNN", command=self.classify_and_update_color
        )
        self.generate_button_classify_dm = tk.Button(
            self.master,text="Classifique apos DM",command=self.classifier_reduction
        )
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        
        self.random_point_index = []
        
        self.generate_button_radom.grid(row=0, column=0)  # Coluna 0
        self.generate_button_fill_matrix.grid(row=0, column=1) # Coluna 1
        self.generate_button_classifier.grid(row=0, column=2)  # Coluna 2
        self.generate_button_random_dm.grid(row=0, column=3) #coluna 3
        self.generate_button_classify_dm.grid(row=0,column=4) #coluna 4
        self.canvas.get_tk_widget().grid(row=1, columnspan=5)  # Span para ocupar 5 colunas
        
    @property
    def get_matrix(self):
        matrix = np.column_stack((self.data,self.target))
        return matrix
    
    def get_plot_area_size(self):
        # Obtém as dimensões atuais do eixo (axes)
        largura = self.ax.get_xlim()[1]
        altura = self.ax.get_ylim()[1]
        return largura, altura


   
    '''Função para plotagem, pode ou não receber pontos extras e os plotar'''
    def plotagem(self):
        
        
        #Vetor com as cores de cada rotulo
        cores = ['salmon','khaki','skyblue']
        
        colors = [cores[int(label)] for label in self.matrix[:,-1]]

        plt.scatter(self.data[:,0],self.data[:,1],color=colors,edgecolors='k',zorder=0)
        self.canvas.draw()
    
    
    #Calculo da matriz de distância
    def calculate_distance_matrix(self,data):
        len_of_matrix = data.shape[0]
        distance_matrix = np.zeros((len_of_matrix,len_of_matrix))
        
        for i in range(len_of_matrix):
            for j in range(len_of_matrix):
                distance_matrix[i,j] = np.linalg.norm(data[i] - data[j])

        return distance_matrix
    
    def fit_transform(self, data):
        try:
            # Separe as labels da matriz
            labels = data[:, -1]
            
            # Remova a última coluna (que contém as labels)
            data = data[:, :-1]
            data = np.asmatrix(data)
            N = data.shape[0]

            # Calcule a distância da matriz D
            D = self.calculate_distance_matrix(data)

            # Calcule o kernel ke da matriz
            ke = np.exp(-D**2 / self.epsilon)

            # Calcule o vetor d
            d = ke.sum(axis=1)

            # Calcule a matriz kea
            kea = ke / (np.outer(d, d) ** self.alpha)

            # Calcule o vetor sqrtPi
            sqrtPi = np.sqrt(kea.sum(axis=1))

            # Calcule a matriz A
            A = kea / np.outer(sqrtPi, sqrtPi)

            U, _, _ = np.linalg.svd(A)

            U_normalized = U / sqrtPi[:, np.newaxis]

            Ut = U_normalized

            # Combine a matriz reduzida com as labels
            reduced_matrix_with_labels = np.column_stack((Ut, labels))

            return reduced_matrix_with_labels

        except ValueError as e:
            return None, str(e)
    
    def fill_the_matrix(self):
        self.ax.clear()
        self.plotagem()
        numero_de_pontos_x = 25
        numero_de_pontos_y = 15

        largura_janela,altura_janela = self.get_plot_area_size()
      
        # Tamanho dos pontos
        tamanho_pontos = min(largura_janela / numero_de_pontos_x, altura_janela / numero_de_pontos_y)

        x = []
        y = []

        for i in range(numero_de_pontos_y):
            for j in range(numero_de_pontos_x):
                x.append(j * tamanho_pontos)
                y.append(i * tamanho_pontos)
                


        # Converta as listas em arrays numpy
        self.value_x = np.array(x)
        self.value_y = np.array(y)
        
        # Plote os pontos
        self.ax.scatter(self.value_x, self.value_y, c='blue', marker='o', edgecolors='k', s=100,zorder=1)
        self.canvas.draw()
        

    def make_random_point(self):
        self.ax.clear()
        #gerar 10 pontos aleatorios
        self.plotagem()
        self.value_x = np.random.uniform(0, 70,5)
        self.value_y = np.random.uniform(0, 70,5)
        self.ax.scatter(self.value_x, self.value_y, color='blue', marker='o', edgecolors='k', s=100,zorder=1)
        self.canvas.draw()
    
    # def train_test_split(self):
    #     X_train,X_test,y_train,y_test = train_test_split(self.data,self.target,test_size=0.2,random_state=42)
    
    def classifier(self):
        model = KNeighborsClassifier()
        model.fit(self.matrix[:,:2],self.matrix[:,-1])
        predict_matrix = np.column_stack((self.value_x, self.value_y))
        predictions = model.predict(predict_matrix)
        self.update_point_color(predictions)  # Chama o método para atualizar a cor
        return predictions
    
    def classifier_reduction(self):
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(self.matrix_reduction[:,1:3], self.matrix_reduction[:, -1])
        predictions = model.predict(np.column_stack((self.value_x, self.value_y)))
        self.update_point_color(predictions)
        return predictions

    def random_reduction_points(self):
        # Gerar 10 pontos aleatórios
        self.value_x = np.random.uniform(0, 70,10)
        self.value_y = np.random.uniform(0, 70,10)
        self.value_z = np.random.uniform(0, 70,10)
        matrix = np.zeros((len(self.value_x),3))
        matrix[:,0] = self.value_x
        matrix[:,1] = self.value_y
        matrix[:,2] = self.value_z
        matrix_reduzida = self.fit_transform(matrix)
        self.value_x = matrix_reduzida[:,1]
        self.value_y = matrix_reduzida[:,2]
        self.plot_with_dimensinal_reduction()  # Certifique-se de que os dados estejam atualizados
        self.ax.scatter(self.value_x, self.value_y, color='blue', marker='o', edgecolors='k', zorder=1)
        self.canvas.draw()

    def plot_with_dimensinal_reduction(self):
        self.ax.clear()
        # Vetor com as cores de cada rótulo
        cores = ['salmon', 'khaki', 'skyblue']
        
        matrix_result = self.fit_transform(self.get_matrix)
        
        self.matrix_reduction = matrix_result
        

        # Lista de cores correspondentes aos rótulos
        colors = [cores[int(label)] for label in matrix_result[:, -1]]
        
        
        self.ax.scatter(matrix_result[:, 1], matrix_result[:, 2], c=colors, edgecolors='k', zorder=0)

        self.canvas.draw()

        
    def update_point_color(self, predictions):
        colors=['red','yellow','seagreen']
        for i, prediction in enumerate(predictions):
            color = colors[int(prediction)]
            self.ax.scatter(self.value_x[i], self.value_y[i], color=color, marker='o',edgecolors='k', s=100,zorder=1)
        self.canvas.draw()
    
    def classify_and_update_color(self):
        prediction = self.classifier()
        self.update_point_color(prediction)
        
        
        
            
        
        