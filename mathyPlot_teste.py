import mathyPlot as mtp
import tkinter as tk


primeiro_rotulo = mtp.Rotulos(60, 30, 36, 144, 5000, 90, 16)
segundo_rotulo = mtp.Rotulos(20, 40, 16, 16, 5000, 60, 16)
terceiro_rotulo = mtp.Rotulos(25, 10, 100, 25, 5000, 5, 9)

root = tk.Tk()

conjunto_rotulos = mtp.Conjunto_rotulos(root, primeiro_rotulo, segundo_rotulo, terceiro_rotulo)

root.mainloop()



