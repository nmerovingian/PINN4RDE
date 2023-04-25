import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd
import seaborn as sns 
import matplotlib.cm as cm


linewidth = 4
fontsize = 15

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs


Scs = np.array([100,200,500,800,1000,1200,1400])
colors = cm.YlGnBu(1.0-(Scs/1400)*0.8)

PINN_files = ['PINN Sc=100\sigma=1.00E-01 Sc= 1.00E+02 epochs=1.00E+02 Lambda_ratio=5.0 n_train=1000000 expansion_terms = 2 dr=0.03 CV.csv',
              "PINN Sc=200\sigma=1.00E-01 Sc= 2.00E+02 epochs=1.00E+02 Lambda_ratio=5.0 n_train=1000000 expansion_terms = 2 dr=0.03 CV.csv",
              "PINN Sc=500\sigma=1.00E-01 Sc= 5.00E+02 epochs=1.00E+02 Lambda_ratio=5.0 n_train=1000000 expansion_terms = 2 dr=0.03 CV.csv",
              "PINN Sc=800\sigma=1.00E-01 Sc= 8.00E+02 epochs=1.00E+02 Lambda_ratio=5.0 n_train=1000000 expansion_terms = 2 dr=0.03 CV.csv",
              "PINN Sc=1000\sigma=1.00E-01 Sc= 1.00E+03 epochs=1.00E+02 Lambda_ratio=5.0 n_train=1000000 expansion_terms = 2 dr=0.03 CV.csv",
              "PINN Sc=1200\sigma=1.00E-01 Sc= 1.20E+03 epochs=1.00E+02 Lambda_ratio=5.0 n_train=1000000 expansion_terms = 2 dr=0.03 CV.csv",
              "PINN Sc=1400\sigma=1.00E-01 Sc= 1.40E+03 epochs=1.00E+02 Lambda_ratio=5.0 n_train=1000000 expansion_terms = 2 dr=0.03 CV.csv"]


fig,ax = plt.subplots(figsize=(8,4.5))


with open('PINN sigma= 0.10 Sc=100\sigma=1.00E-01 Sc= 1.00E+02 epochs=1.00E+02 Lambda_ratio=5.0 n_train=1000000 expansion_terms = 0 dr=0.03.npy','rb') as f:
    Data = np.load(f)
    T_0,W_0,C_0 = Data[0],Data[1],Data[2]

with open('PINN sigma= 0.10 Sc=100\sigma=1.00E-01 Sc= 1.00E+02 epochs=1.00E+02 Lambda_ratio=5.0 n_train=1000000 expansion_terms = 2 dr=0.03.npy','rb') as f:
    Data = np.load(f)
    T_2,W_2,C_2 = Data[0],Data[1],Data[2]

axes = ax.pcolormesh(T_0,W_0,C_2-C_0,cmap='YlGnBu',shading='auto')

ax.set_xlabel(r'T',fontsize='large',fontweight='bold')
ax.set_ylabel(r'W',fontsize='large',fontweight='bold')

cbar = plt.colorbar(axes,pad=0.05, aspect=10,ax=ax)

cbar.set_label(r'$C_{n=2} - C_{n=0}$',fontsize='large',fontweight='bold')





fig.savefig('Different Sc.png',dpi=250,bbox_inches='tight')