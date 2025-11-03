import numpy as np

rh = 2.0 #AU
rmin = 0.91 #AU
arcs = np.linspace(rh,rmin,5)
arcs2 = np.arange(rh,rmin,-0.25)
print(arcs,arcs2)