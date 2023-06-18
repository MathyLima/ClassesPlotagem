import mathyPlot as mtp
import numpy as np

primeiro_rotulo = mtp.Rotulos(60,30,36,144,100000)
segundo_rotulo = mtp.Rotulos(20,40,16,16,100000)
terceiro_rotulo = mtp.Rotulos(25,10,100,25,150005)

conjunto_rotulos = mtp.Conjunto_rotulos(primeiro_rotulo,segundo_rotulo,terceiro_rotulo)

conjunto_rotulos.plotagem([10,40,60],[60,100,30])



