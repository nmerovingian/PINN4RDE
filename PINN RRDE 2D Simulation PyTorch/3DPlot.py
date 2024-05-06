from matplotlib import pyplot as plt 
import numpy as np
from matplotlib import cm  


linewidth = 4
fontsize = 15
font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs

colors = cm.Set2(np.linspace(0,1,8))

X_cor = []
Y_cor = []
Z_cor = []

fig = plt.figure(figsize=(8,4.5))
ax = plt.axes(projection='3d')
X = np.logspace(-4,-2,num=10000,endpoint=True) # disk radius, m 
Y = np.linspace(1,150,num=100,endpoint=True) # Frequencies, Hz
Z = np.linspace(100,1500,num=100,endpoint=True) # Sc values 

X,Y,Z = np.meshgrid(X,Y,Z)

Re = X**2*Y*np.pi*2/1e-6  # Reynold number
C = Z**(1/3)*Re**(1/2)

X = X.reshape(-1)
Y = Y.reshape(-1)
Z = Z.reshape(-1)
C = C.reshape(-1)



for index in range(len(X)):
    if 12.1>C[index]>11.9:
        X_cor.append(X[index])
        Y_cor.append(Y[index])
        Z_cor.append(Z[index])


print(X_cor)


X_cor = [value*1000 for value in X_cor]
ax.plot_trisurf(X_cor,Y_cor,Z_cor,antialiased = True,facecolor=tuple(colors[4]),alpha=0.8,shade=True,linewidth=0)

ax.set_xlabel(r'$r_1, \ mm$',fontsize="large",fontweight='bold',labelpad=5)
ax.set_ylabel(r'$freq, \ Hz$',fontsize="large",fontweight='bold',labelpad=5)
ax.set_zlabel(r'$Sc$',fontsize="large",fontweight='bold',labelpad=-34)

plt.tight_layout()
ax.view_init(29,29,0)
fig.savefig('./analysis/Experiment.png',dpi=250,bbox_inches='tight')
fig.savefig('E:\OneDrive - Nexus365\Project PINN Hydrodynamic\Paper 15 Figures\Figure 6.png',dpi=250,bbox_inches='tight')