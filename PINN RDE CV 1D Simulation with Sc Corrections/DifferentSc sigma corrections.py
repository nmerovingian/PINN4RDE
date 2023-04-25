import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd
import seaborn as sns 
import matplotlib.cm as cm


linewidth = 2
fontsize = 9

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs


Scs = [100,200,500,800,1000,1200,1400]
sigmas = [0.1,0.5]
n_corrs = [0,1,2,6]
colors = cm.YlGnBu(1.0-(np.array(Scs)/1400)*0.7)[::-1]

fig,axs = plt.subplots(figsize=(8,12),nrows=4,ncols=2)

for col_index,sigma in enumerate(sigmas):
    for row_index,n_corr in enumerate(n_corrs):
        ax = axs[row_index][col_index]
        axins = ax.inset_axes([0.75,0.1,0.2,0.6])
        df_FD = pd.read_csv(f'FD Simulation\sigma={sigma:.6E}.csv')

        for index,Sc in enumerate(Scs):
            df = pd.read_csv(f'PINN sigma= {sigma:.2f} Sc={Sc:.0f}\sigma={sigma:.2E} Sc= {Sc:.2E} epochs=1.00E+02 Lambda_ratio=5.0 n_train=1000000 expansion_terms = {n_corr:.0f} dr=0.03 CV.csv')
            ax.plot(df.iloc[:,0],df.iloc[:,1],color=tuple(colors[index]),lw=2,label=f'Sc={Scs[index]}',alpha=0.85)
            axins.plot(df.iloc[:,0],df.iloc[:,1],color=tuple(colors[index]),lw=2,label=f'Sc={Scs[index]}')


        ax.plot(df_FD.iloc[:,0],df_FD.iloc[:,1],ls='--',color='r',lw=1.5,alpha=0.5)
        axins.plot(df_FD.iloc[:,0],df_FD.iloc[:,1],ls='--',color='r',lw=1.5,alpha=0.5)

        axins.set_xlim(-9.99,-9)
        axins.set_ylim(-1.01,-0.91)
        axins.set_xticklabels([])
        axins.set_xticks([])
        ax.indicate_inset_zoom(axins, edgecolor="black",lw=2)
        ax.legend(title=f'$\sigma={sigma:.1f} \ \ n_{{corr}}={n_corr}$',loc=2,fontsize=8)

        if col_index == 0:
            ax.set_ylabel(r'$J$',fontsize='large')
        if row_index == 3:
            ax.set_xlabel(r'$\theta$',fontsize='large')


fig.text(0.05,0.88,'(a)',fontsize=12,fontweight='bold')
fig.text(0.05,0.68,'(b)',fontsize=12,fontweight='bold')
fig.text(0.05,0.48,'(c)',fontsize=12,fontweight='bold')
fig.text(0.05,0.28,'(d)',fontsize=12,fontweight='bold')


fig.text(0.48,0.88,'(e)',fontsize=12,fontweight='bold')
fig.text(0.48,0.68,'(f)',fontsize=12,fontweight='bold')
fig.text(0.48,0.48,'(g)',fontsize=12,fontweight='bold')
fig.text(0.48,0.28,'(h)',fontsize=12,fontweight='bold')


fig.savefig('Change Sigma n_corr Sc.png',dpi=250,bbox_inches='tight')






