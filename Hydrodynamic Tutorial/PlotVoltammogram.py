from matplotlib import pyplot as plt 
import pandas as pd
import numpy as np
from matplotlib.cm import viridis


linewidth = 4
fontsize = 15

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs

#Change the range of scan rates to be plotted
sigmas = [0.001,0.01,0.1,0.5,1,10,30,100,1000][:3]

colors = viridis(np.linspace(0,1,len(sigmas)))

directory = f'FD Simulation Results'
fig, ax = plt.subplots(figsize=(8,4.5))
for index,sigma in enumerate(sigmas):
    file_name = f'{directory}/sigma={sigma:.6E}.csv'
    df = pd.read_csv(file_name)

    ax.plot(df.iloc[:,0],df.iloc[:,1],label=f'$\sigma={sigma:.2E}$',color = tuple(colors[index]))


ax.legend(fontsize=12)
ax.set_xlabel(r'Dimensionless Potential, $\theta$',fontsize='large',fontweight='bold')
ax.set_ylabel(r'Dimensionless Flux, $J$')


fig.savefig("./Figures/CV.png",dpi=250,bbox_inches='tight')

fig, ax = plt.subplots(figsize=(8,4.5))
for index,sigma in enumerate(sigmas):
    file_name = f'{directory}/sigma={sigma:.6E}.csv'
    df = pd.read_csv(file_name)

    df.iloc[:,1] = df.iloc[:,1]/np.sqrt(sigma)

    ax.plot(df.iloc[:,0],df.iloc[:,1],label=f'$\sigma={sigma:.2E}$',color = tuple(colors[index]))


ax.legend(fontsize=12)
ax.set_xlabel(r'Dimensionless Potential, $\theta$',fontsize='large',fontweight='bold')
ax.set_ylabel(r'Normalized Dimensionless Flux, $\frac{J}{\sqrt{\sigma}}$')


fig.savefig("./Figures/CV-Normalized.png",dpi=250,bbox_inches='tight')


