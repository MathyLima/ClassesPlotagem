import mathyPlot as mtp
import tkinter as tk
from sklearn import datasets

primeiro_rotulo = mtp.Rotulos(60, 30, 36, 144, 300, 90, 16)
segundo_rotulo = mtp.Rotulos(20, 40, 16, 16, 300, 60, 16)
terceiro_rotulo = mtp.Rotulos(25, 10, 100, 25, 300, 5, 9)

iris = datasets.load_iris()
data = iris.data
target = iris.target
print(data)

root = tk.Tk()

conjunto_rotulos = mtp.Join_to_matrix(primeiro_rotulo, segundo_rotulo, terceiro_rotulo)

# data = conjunto_rotulos.get_matrix[:,0:2]

# target = conjunto_rotulos.get_matrix[:,-1]

plotagem = mtp.Plot_data(root,data,target,epsilon=35,alpha=1,dimension=2)


root.mainloop()



