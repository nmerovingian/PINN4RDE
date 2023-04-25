import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd
import seaborn as sns 
import matplotlib.cm as cm
from Helper import calcW

FD_directory = 'Paper Data Change Sigmas'
PINN_data_directory = 'Data'

linewidth = 4
fontsize = 15

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs


#sigmas = [1,2,5,10,15,20,30,40,50,60,70,80,90,100]
sigmas = [0.1,1,5,10,30,50,100]

fig,axs = plt.subplots(figsize=(8,13.5),nrows=3)

colors = cm.Blues(np.linspace(0.2,1,len(sigmas)))






freq = 50 #Hz
Omega = 50 * np.pi*2 # rad/sec
nu = 1e-6 #m^2 s^-1 kinematic viscosity
A_factor =  0.51023*Omega**1.5*nu**-0.5 
re = 1e-5 # m, radius of electrode 
D = 1e-9 # m^2 s^-1, diffusion coefficients 
script_A_factor = A_factor * re**3 / D
FluxConvertToHale = np.sqrt(1.65894)/(((A_factor/D)**(1/3))*re)
TimeConvertToHale = (A_factor/D)**(2/3)*(re**2)






ax = axs[0]

for index, sigma in enumerate(sigmas):
    FD_file = f'{FD_directory}/sigma={sigma:.6E}.csv'
    PINN_file = f'{PINN_data_directory}/CV sigma={sigma:.2E} epochs=1.50E+02 n_train=1.00E+06.csv'

    df_FD = pd.read_csv(FD_file)
    df_PINN = pd.read_csv(PINN_file)
    if index == 0:
        ax.plot(df_PINN.iloc[:,0],df_PINN.iloc[:,1],color=tuple(colors[index]),lw=2.5,alpha=0.7,label=f'$\sigma={sigma:.1f}$')
    else:
        ax.plot(df_PINN.iloc[:,0],df_PINN.iloc[:,1],color=tuple(colors[index]),lw=2.5,alpha=0.7,label=f'$\sigma={sigma:.0f}$')
    ax.plot(df_FD.iloc[:,0],df_FD.iloc[:,1],color=tuple(colors[index]),lw=2.5,alpha=0.7,ls='--')


axin = ax.inset_axes([0.65,0.2,0.33,0.33])
df_SS = pd.read_csv('Strutwolf and Schoeller.csv')
axin.plot(np.sqrt((df_SS.iloc[:,0])**2/2.5921*1.566),-df_SS.iloc[:,1],label='S&S',color='r')
df_PINN_Peak = pd.read_csv('PINN Peak Fluxes.csv')
df_PINN_Peak = df_PINN_Peak[df_PINN_Peak['sigma']<65]
axin.scatter(np.sqrt(df_PINN_Peak.iloc[:,0]),df_PINN_Peak.iloc[:,1],s=20,label='PINN')


axin.tick_params(axis='both', which='major', labelsize='small')
axin.set_xlabel(r'$\sqrt{\sigma}$',fontsize='small')
axin.set_ylabel(r'$J_p$',fontsize='small')
axin.legend(fontsize='small',labelspacing=0.2,frameon=False)

ax.set_xlabel(r'$\theta$',fontsize='large')
ax.set_ylabel(r'$J$',fontsize='large')

ax.legend(ncol=2,loc='upper left',fontsize='small')


ax = axs[1]
with open('Data\sigma=4.00E+01 epochs=1.50E+02 n_train=1.00E+06.npy','rb') as f:
    Data = np.load(f)

T,Y,C = Data[0],Data[1],Data[2]
W = calcW(Y)
axes = ax.pcolormesh(T,W,C,cmap='YlGnBu',shading='auto')
ax.set_xlabel(r'$T$',fontsize='large',fontweight='bold')
ax.set_ylabel(r'$W$',fontsize='large',fontweight='bold')
cbar = plt.colorbar(axes,pad=0.05, aspect=10,ax=ax)
ax.set_ylim(0,2)
cbar.set_label(r'$C(T,Y)$',fontsize='large',fontweight='bold')


ax = axs[2]
df_PINN = pd.read_csv('chronoamperogram maxT=1.00E+00 epochs=4.00E+01 n_train=1000000.csv')

df_BP = pd.read_csv('Bruckenstein & Prager.csv',header=None) 
df_BP.iloc[:,1] = -df_BP.iloc[:,1]
df_BP.iloc[:,0] += (0.1*TimeConvertToHale)
ax.plot(df_PINN.iloc[:,0],df_PINN.iloc[:,1],color='b',lw=3,label='PINN',alpha=0.7)
ax.scatter(df_BP.iloc[:,0],df_BP.iloc[:,1],color='r',s=40,label='Bruckenstein & Prager')

ax.set_xlabel(r'$T$',fontsize='large',fontweight='bold')
ax.set_ylabel(r'$J$',fontsize='large',fontweight='bold')
ax.legend()
fig.text(0.03,0.89,'(a)',fontsize=20,fontweight='bold')
fig.text(0.03,0.63,'(b)',fontsize=20,fontweight='bold')
fig.text(0.03,0.33,'(c)',fontsize=20,fontweight='bold')
fig.savefig('1D Simulation.png',dpi=250,bbox_inches='tight')

