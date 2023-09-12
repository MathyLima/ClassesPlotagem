import mathyPlot as mtp
import tkinter as tk


primeiro_rotulo = mtp.Rotulos(60, 30, 36, 144, 300, 90, 16)
segundo_rotulo = mtp.Rotulos(20, 40, 16, 16, 300, 60, 16)
terceiro_rotulo = mtp.Rotulos(25, 10, 100, 25, 300, 5, 9)

root = tk.Tk()

conjunto_rotulos = mtp.Conjunto_rotulos(root, primeiro_rotulo, segundo_rotulo, terceiro_rotulo,epsilon=35,alpha=1,dimension=2)

root.mainloop()



