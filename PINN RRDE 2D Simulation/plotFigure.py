from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from NAnalytical import calcN_AB,calcN_IL
import matplotlib.ticker as mtick
from matplotlib import cm 
from scipy.optimize import curve_fit
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
colors = cm.Set2(np.linspace(0,1,8))

def func(x,a,b,c):
    return 0.4002 * (1.0 + a*(1.0/x) + b*((1.0/x)**2)+ c*((1.0/x)**3))
def to_axis_1(x):
    return x

def to_axis_2(x):
    r1 = x*1e-6
    Sc = nu/D # Schmidt number
    Re = r1**2*omega/nu  # Reynold number

    return Sc**(1/3)*Re**(1/2)

def to_radius(x):
    return x/((nu/D)**(1/3)*(2*np.pi*freq/nu)**(1/2))*1e6

linewidth = 4
fontsize = 15
font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs

radius_multiple_sweep = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0]


r1s = []
Xs = []
JD00s = []
JD01s = []
JD10s = []
JD11s = []
JD12s = []
JD21s = []
JD22s = []

JD_Levichs = []

JR00s = []
JR01s = []
JR10s = []
JR11s = []
JR12s = []
JR21s = []
JR22s = []

N00s = []
N01s = []
N10s = []
N11s = []
N12s = []
N21s = []
N22s = []
NABs = []


r1s_NR = []
Xs_NR = []

JD00s_NR = []
JD01s_NR = []
JD10s_NR = []
JD11s_NR = []
JD12s_NR = []
JD21s_NR = []
JD22s_NR = []

JD_Levichs_NR = []

JR00s_NR = []
JR01s_NR = []
JR10s_NR = []
JR11s_NR = []
JR12s_NR = []
JR21s_NR = []
JR22s_NR = []

N00s_NR= []
N01s_NR= []
N10s_NR= []
N11s_NR= []
N12s_NR= []
N21s_NR= []
N22s_NR= []


for ID,radius_multiple in enumerate(radius_multiple_sweep):
    if ID>0:

        r1 = 1e-5 * radius_multiple
        r2 = 1.1e-5 * radius_multiple
        r3 = 1.44e-5 * radius_multiple

        R1 = r1/r1 # R1 is the dimnesionless radius of disk electrode, should always be 1. 
        R2 = r2/r1
        R3 = r3/r1

        freq = 5 #Hz
        omega = freq * np.pi*2 # Rotational freq
        nu = 1e-6 #s^-1 kinematic viscosity
        L =  0.51023*omega**1.5*nu**(-0.5)


        D = 1e-9 # m^2 s^-1, diffusion coefficients 

        scriptL = L * r1**3 / D # The dimensionless form of L


        Sc = nu/D # Schmidt number
        Re = r1**2*omega/nu  # Reynold number

        FluxConvertToHale = np.sqrt(1.65894)/(((L/D)**(1/3))*r1) # Flux converted using Hale Transformation
        TimeConvertToHale = (L/D)**(2/3)*(r1**2)
        SigmaConvertToHale = (D/L)**(2/3)/(r1**2)
        delta = 1.2865*(D/L)**(1/3) # diffusion layer thickness, m 
        Delta = delta/r1 # dimensionless diffusion layer thickness
        deltaH = np.sqrt(nu/omega) #hydrodynamic layer thickness 
        DeltaH =  deltaH/r1 #Dimensionless hydrodynamic layer thickness

        collection_efficiency_AB = calcN_AB(R1,R2,R3)
        collection_efficiency_IL = calcN_IL(R1,R2,R3)


        if ID<10:
            saving_directory = f"Round 10 ID = {ID} epochs = 300 lambda_ratio = 3.00 radius_multiple = {radius_multiple:.2f} No Flux = False RadialD = True"
            saving_directory_NR = f"Round 9 ID = {ID} epochs = 300 lambda_ratio = 3.00 radius_multiple = {radius_multiple:.2f} No Flux = False RadialD = False"
        else:
            saving_directory = f"Round 10h ID = {ID-10} epochs = 300 lambda_ratio = 3.00 radius_multiple = {radius_multiple:.2f} No Flux = False RadialD = True"
            saving_directory_NR = f"Round 9h ID = {ID-10} epochs = 300 lambda_ratio = 3.00 radius_multiple = {radius_multiple:.2f} No Flux = False RadialD = False"


        df = pd.read_csv(f'{saving_directory}/results.csv')
        df_NR = pd.read_csv(f"{saving_directory_NR}/results.csv")

        r1s.append(r1*1e6)
        r1s_NR.append(r1*1e6)
        Xs.append(df['X'].iloc[0])
        Xs_NR.append(df_NR['X'].iloc[0])
        JD00s.append(df['J_D'].iloc[0])
        JD_Levichs.append(df['J_D_Levichs'].iloc[0])
        JR00s.append(df['J_R'].iloc[0])
        N00s.append(df['collection efficiency'].iloc[0])
        NABs.append(calcN_AB(r1,r2,r3))

        N01s.append(df['collection efficiency'].iloc[1])
        JD01s.append(df['J_D'].iloc[1])
        JR01s.append(df['J_R'].iloc[1])

        N10s.append(df['collection efficiency'].iloc[3])
        JD10s.append(df['J_D'].iloc[3])
        JR10s.append(df['J_R'].iloc[3])


        N11s.append(df['collection efficiency'].iloc[5])
        JD11s.append(df['J_D'].iloc[5])
        JR11s.append(df['J_R'].iloc[5])


        N12s.append(df['collection efficiency'].iloc[6])
        JD12s.append(df['J_D'].iloc[6])
        JR12s.append(df['J_R'].iloc[6])

        N21s.append(df['collection efficiency'].iloc[8])
        JD21s.append(df['J_D'].iloc[8])
        JR21s.append(df['J_R'].iloc[8])

        N22s.append(df['collection efficiency'].iloc[10])
        JD22s.append(df['J_D'].iloc[10])
        JR22s.append(df['J_R'].iloc[10])



        JD00s_NR.append(df_NR['J_D'].iloc[0])
        JR00s_NR.append(df_NR['J_R'].iloc[0])
        N00s_NR.append(df_NR['collection efficiency'].iloc[0])
        JD_Levichs_NR.append(df_NR['J_D_Levichs'].iloc[0])

fig,axs = plt.subplots(figsize=(8,13.5),nrows=3)

ax = axs[0]

colors = cm.tab20(np.linspace(0,1,20))
ax.bar(Xs,[JD+JR for JD, JR in zip(JD00s,JR00s)],bottom = -np.array(JR00s),label=r'$J_D$, w/ Rad. Diff.',alpha=1.0,color=tuple(colors[1]),width=0.15)
ax.bar(Xs,-np.array(JR00s),label=r'$|J_R|$, w/ Rad. Diff.',alpha=1.0,color=tuple(colors[0]),width=0.15)
ax.bar(np.array(Xs_NR)+0.18,[JD+JR for JD,JR in zip(JD00s_NR,JR00s_NR)],bottom=-np.array(JR00s_NR),label=r'$J_D$, w/o Rad. Diff.',alpha=1.0,color=tuple(colors[3]),width=0.15)
ax.bar(np.array(Xs_NR)+0.18,-np.array(JR00s_NR),label=r'$|J_R|$, w/o Rad. Diff.',color=tuple(colors[2]),width=0.15)
#ax.bar(np.array(Xs_NR)+0.3,JD_Levichs_NR,label='$J_D$ Levich',color=tuple(colors[5]),width=0.15)

ax.set_ylabel('Flux',fontsize='large',fontweight='bold')
ax.legend(fontsize=12)
ax.xaxis.set_minor_locator(AutoMinorLocator())
sec_ax = ax.secondary_xaxis(-0.2,functions=(to_radius,to_axis_1))
sec_ax.set_xlabel(r'$r_1, \mu m$ ',fontsize='large',fontweight='bold')
#ax.set_xlabel(r'$Sc^{\frac{1}{3}}Re^{\frac{1}{2}}$',fontsize='large',fontweight='bold')


ax = axs[1]


ax.plot(Xs,np.array(JD00s)/np.array(JD00s_NR),marker='o',label=r'$\frac{J_D, w/ \ \ Rad.\ Diff.}{J_D, w/o \ \ Rad.\ Diff.}$',color=tuple(colors[4]))
ax.plot(Xs,np.array(JR00s)/np.array(JR00s_NR),marker='o',label=r'$\frac{J_R, w/ \ \ Rad.\ Diff.}{J_R, w/o \ \ Rad.\ Diff.}$',color=tuple(colors[5]))
df_deviation = pd.DataFrame({'X':Xs,"J_D":np.array(JD00s)/np.array(JD00s_NR),"J_R":np.array(JR00s)/np.array(JR00s_NR)})
df_deviation.to_csv('./analysis/deviation.csv',index=False)
ax.axhline(y=1,ls='--',color='k',lw=2,alpha=0.6)
ax.set_ylabel(r'Percentage')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.legend()



ax = axs[2]
ax.plot(Xs,N00s,marker='o',alpha=0.8,color='k',label=f'$N$ w/ Rad. Diff.',lw=2)
ax.axhline(y=0.4002,color='k',ls='--',label=r'$N_{AB}$')
ax.axhline(y=calcN_IL(R1,R2,R3),color='k',ls='-.',label=r'$N_{IL}$')
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
#sigma = 1.0/(np.array(N00s))
popt,pconv = curve_fit(func,Xs,N00s,absolute_sigma=False)
Xmin = Xs[0]
Xmax = Xs[-1]
#ax.plot(np.linspace(Xmin,Xmax),func(np.linspace(Xmin,Xmax),*popt),label=f'$Y=N_{{AB}}(1{popt[0]:+.2f}x^{{-1}}{popt[1]:+.2f}x^{{-2}}$\n${popt[2]:+.2f}x^{{-3}})$',lw=3,alpha=1.0,color=tuple(colors[6]))
ax.legend(fontsize=12)
ax.set_xlabel(r'$Sc^{\frac{1}{3}}Re^{\frac{1}{2}}$',fontsize='large',fontweight='bold')
sec_ax = ax.secondary_xaxis(-0.2,functions=(to_radius,to_axis_1))
sec_ax.set_xlabel(r'$r_1, \mu m$ ',fontsize='large',fontweight='bold')


fig.text(0.05,0.88,'a)',fontsize=20)
fig.text(0.05,0.63,'b)',fontsize=20)
fig.text(0.05,0.33,'c)',fontsize=20)


fig.savefig(f'./analysis/Round 10 Single.png',dpi=250,bbox_inches='tight')
fig.savefig(f'E:\OneDrive - Nexus365\Project PINN Hydrodynamic\Paper 15 Figures\Figure 4.png',dpi=250,bbox_inches='tight')




##################################################################################
fig,axs = plt.subplots(figsize=(24,12),nrows=2,ncols=3)

ax = axs[0][0]

ax.plot(Xs,np.array(JD00s)/np.array(JD_Levichs),marker='o',label=r'$J_D/J_{D,Levich}$')
ax.axhline(y=1,ls='--',color='r',lw=3)
ax.set_ylabel(r'$J_D/J_{D,Levich}$')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend()

ax = axs[0][1]
ax.plot(Xs,np.array(JD00s)/np.array(JD00s_NR),marker='o',label=r'$J_D/J_{D, No Radial}$')
ax.axhline(y=1,ls='--',color='r',lw=3)
ax.set_ylabel(r'$J_D/J_{D, No Radial}$')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend()


ax = axs[0][2]
J_R_Levichs = np.array(JD_Levichs)*np.array(NABs)
ax.plot(Xs,-np.array(JR00s)/J_R_Levichs,marker='o',label=r'$J_R/J_{R,Levichs}$')
ax.axhline(y=1,ls='--',color='r',lw=3)
ax.set_ylabel(r'$J_R/J_{R,Levichs}$')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend()


ax = axs[1][0] 
ax.plot(Xs,np.array(JR00s)/np.array(JR00s_NR),marker='o',label=r'$J_R/J_{R, No Radial}$')
ax.axhline(y=1,ls='--',color='r',lw=3)
ax.set_ylabel(r'$J_R/J_{R, No Radial}$')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend()
ax.set_xlabel(r'$Sc^{\frac{1}{3}}Re^{\frac{1}{2}}$',fontsize='large',fontweight='bold')
sec_ax = ax.secondary_xaxis(-0.2,functions=(to_radius,to_axis_1))
sec_ax.set_xlabel(r'$r_1, \mu m$ ',fontsize='large',fontweight='bold')

ax = axs[1][1]
ax.plot(Xs,N00s,label=r'$N$ with Radial',marker='o')
ax.plot(Xs,N00s_NR,label=r'$N$ No Radial',marker='o')
ax.plot(Xs,NABs,label=r'$N_{AB}$',marker='*')
ax.set_ylabel('Collection Efficiency')
ax.legend()
ax.set_xlabel(r'$Sc^{\frac{1}{3}}Re^{\frac{1}{2}}$',fontsize='large',fontweight='bold')
sec_ax = ax.secondary_xaxis(-0.2,functions=(to_radius,to_axis_1))
sec_ax.set_xlabel(r'$r_1, \mu m$ ',fontsize='large',fontweight='bold')

ax = axs[1][2]
ax.set_xlabel(r'$Sc^{\frac{1}{3}}Re^{\frac{1}{2}}$',fontsize='large',fontweight='bold')


fig.savefig('./analysis/Richard Results.png',dpi=250,bbox_inches='tight')

####################################################################################




fig,ax = plt.subplots(figsize=(16,9))
#twin1 = ax.twinx()

ax.plot(Xs,N00s,marker='o',alpha=0.8,color='k',label=f'$n_{{corr,R}}=0$ $n_{{corr,Y}}=0$',lw=2)
ax.plot(Xs,N01s,marker='o',alpha=0.8,label=f'$n_{{corr,R}}=0$ $n_{{corr,Y}}=1$',lw=2)
ax.plot(Xs,N10s,marker='o',alpha=0.8,label=f'$n_{{corr,R}}=1$ $n_{{corr,Y}}=0$',lw=2)
ax.plot(Xs,N11s,marker='o',alpha=0.8,label=f'$n_{{corr,R}}=1$ $n_{{corr,Y}}=1$',lw=2)
ax.plot(Xs,N12s,marker='o',alpha=0.8,label=f'$n_{{corr,R}}=1$ $n_{{corr,Y}}=2$',lw=2)
ax.plot(Xs,N21s,marker='o',alpha=0.8,label=f'$n_{{corr,R}}=2$ $n_{{corr,Y}}=1$',lw=2)
ax.plot(Xs,N22s,marker='o',alpha=0.8,label=f'$n_{{corr,R}}=2$ $n_{{corr,Y}}=2$',lw=2)
#twin1.scatter(Xs_NR,N00s_NR,marker='o',alpha=0.8,color='b',label='w/o Radial Diffusion')
ax.legend()
"""
coeffs = np.polyfit(Xs,N0s,deg=2)
p3 = np.poly1d(coeffs)
print(ID,coeffs)
twin1.plot(Xs,p3(Xs),label='w/ Radial Diffusion',lw=3,alpha=1.0,color=tuple(colors[2]))
"""
sigma = 1.0/(np.array(N00s))

popt,pconv = curve_fit(func,Xs,N00s,absolute_sigma=False)

Xmin = Xs[0]
Xmax = Xs[-1]



#twin1.plot(np.linspace(Xmin,Xmax),func(np.linspace(Xmin,Xmax),*popt),label=f'$Y=N_{{AB}}(1{popt[0]:+.2f}x^{{-1}}{popt[1]:+.2f}x^{{-2}}$\n${popt[2]:+.2f}x^{{-3}})$',lw=3,alpha=1.0,color=tuple(colors[6]))
ax.axhline(y=calcN_AB(r1,r2,r3),color=tuple(colors[7]),ls='--',lw=2,label=r'$N_{AB}$')

#twin1.plot(Xs_NR,N1s_NR,marker='o',label='w/o Radial Diffusion',alpha=0.7,color='r',ls='-.')
ax.tick_params(axis='y',colors=tuple(colors[7]))
ax.set_ylabel('Collection\nEfficiency',fontsize='large',fontweight='bold',color=tuple(colors[7]))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
#ax.set_ylim(0.39,0.7)
#ax.set_xlim(1,6)

ax.legend(loc='upper right')

sec_ax = ax.secondary_xaxis(-0.2,functions=(to_radius,to_axis_1))
sec_ax.set_xlabel(r'$r_1, \mu m$ ',fontsize='large',fontweight='bold')
ax.set_xlabel(r'$Sc^{\frac{1}{3}}Re^{\frac{1}{2}}$',fontsize='large',fontweight='bold')
ax.set_xlabel(r'$Sc^{\frac{1}{3}}Re^{\frac{1}{2}}$',fontsize='large',fontweight='bold')
fig.savefig(f'./analysis/Round 10 N.png',dpi=250,bbox_inches='tight')
plt.close('all')



fig,ax = plt.subplots(figsize=(16,9))
#twin1 = ax.twinx()

ax.plot(Xs,N00s,marker='o',alpha=0.8,color='k',label=f'$n_{{corr,R}}=0$ $n_{{corr,Y}}=0$',lw=2)

ax.plot(Xs,N11s,marker='o',alpha=0.8,label=f'$n_{{corr,R}}=1$ $n_{{corr,Y}}=1$',lw=2)

ax.plot(Xs,N22s,marker='x',alpha=0.8,label=f'$n_{{corr,R}}=2$ $n_{{corr,Y}}=2$',lw=2)


#twin1.scatter(Xs_NR,N00s_NR,marker='o',alpha=0.8,color='b',label='w/o Radial Diffusion')
ax.legend()
"""
coeffs = np.polyfit(Xs,N0s,deg=2)
p3 = np.poly1d(coeffs)
print(ID,coeffs)
twin1.plot(Xs,p3(Xs),label='w/ Radial Diffusion',lw=3,alpha=1.0,color=tuple(colors[2]))
"""
sigma = 1.0/(np.array(N00s))

popt,pconv = curve_fit(func,Xs,N00s,absolute_sigma=False)

Xmin = Xs[0]
Xmax = Xs[-1]



#twin1.plot(np.linspace(Xmin,Xmax),func(np.linspace(Xmin,Xmax),*popt),label=f'$Y=N_{{AB}}(1{popt[0]:+.2f}x^{{-1}}{popt[1]:+.2f}x^{{-2}}$\n${popt[2]:+.2f}x^{{-3}})$',lw=3,alpha=1.0,color=tuple(colors[6]))
ax.axhline(y=calcN_AB(r1,r2,r3),color=tuple(colors[7]),ls='--',lw=2,label=r'$N_{AB}$')

#twin1.plot(Xs_NR,N1s_NR,marker='o',label='w/o Radial Diffusion',alpha=0.7,color='r',ls='-.')
ax.tick_params(axis='y',colors=tuple(colors[7]))
ax.set_ylabel('Collection\nEfficiency',fontsize='large',fontweight='bold',color=tuple(colors[7]))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
#ax.set_ylim(0.39,0.7)
#ax.set_xlim(1,6)

ax.legend(loc='upper right')

sec_ax = ax.secondary_xaxis(-0.2,functions=(to_radius,to_axis_1))
sec_ax.set_xlabel(r'$r_1, \mu m$ ',fontsize='large',fontweight='bold')
ax.set_xlabel(r'$Sc^{\frac{1}{3}}Re^{\frac{1}{2}}$',fontsize='large',fontweight='bold')
ax.set_xlabel(r'$Sc^{\frac{1}{3}}Re^{\frac{1}{2}}$',fontsize='large',fontweight='bold')
fig.savefig(f'./analysis/Round 10 N 11.png',dpi=250,bbox_inches='tight')
plt.close('all')


###################################################################################################
colors = cm.Set2(np.linspace(0,1,8))





fig,axs = plt.subplots(figsize=(8,13.5),nrows=3)
ax = axs[0]
ax.plot(Xs,JD00s,marker='o',alpha=0.8,color='k',label=f'$n_{{corr,R}}=0$ $n_{{corr,Y}}=0$',lw=2)
ax.plot(Xs,JD01s,marker='*',alpha=0.8,label=f'$n_{{corr,R}}=0$ $n_{{corr,Y}}=1$',lw=2,color=tuple(colors[0]))
ax.plot(Xs,JD10s,marker='*',alpha=0.8,label=f'$n_{{corr,R}}=1$ $n_{{corr,Y}}=0$',lw=2,color=tuple(colors[1]))

ax.plot(Xs,JD11s,marker='o',alpha=0.8,label=f'$n_{{corr,R}}=1$ $n_{{corr,Y}}=1$',lw=2,color=tuple(colors[2]))
ax.plot(Xs,JD22s,marker='o',alpha=0.8,label=f'$n_{{corr,R}}=2$ $n_{{corr,Y}}=2$',lw=2,color=tuple(colors[3]))
ax.set_ylabel(r'$J_D$',fontsize='large',fontweight='bold')


ax = axs[1]
ax.plot(Xs,JR00s,marker='o',alpha=0.8,color='k',label=f'$n_{{corr,R}}=0$ $n_{{corr,Y}}=0$',lw=2)
ax.plot(Xs,JR01s,marker='*',alpha=0.8,label=f'$n_{{corr,R}}=0$ $n_{{corr,Y}}=1$',lw=2,color=tuple(colors[0]))
ax.plot(Xs,JR10s,marker='*',alpha=0.8,label=f'$n_{{corr,R}}=1$ $n_{{corr,Y}}=0$',lw=2,color=tuple(colors[1]))

ax.plot(Xs,JR11s,marker='o',alpha=0.8,label=f'$n_{{corr,R}}=1$ $n_{{corr,Y}}=1$',lw=2,color=tuple(colors[2]))
ax.plot(Xs,JR22s,marker='o',alpha=0.8,label=f'$n_{{corr,R}}=2$ $n_{{corr,Y}}=2$',lw=2,color=tuple(colors[3]))
ax.set_ylabel(r'$J_R$',fontsize='large',fontweight='bold')


ax = axs[2]
ax.plot(Xs,N00s,marker='o',alpha=0.8,color='k',label=f'$n_{{corr,R}}=0$ $n_{{corr,Y}}=0$',lw=2)
ax.plot(Xs,N01s,marker='*',alpha=0.8,label=f'$n_{{corr,R}}=0$ $n_{{corr,Y}}=1$',lw=2,color=tuple(colors[0]))
ax.plot(Xs,N10s,marker='*',alpha=0.8,label=f'$n_{{corr,R}}=1$ $n_{{corr,Y}}=0$',lw=2,color=tuple(colors[1]))

ax.plot(Xs,N11s,marker='o',alpha=0.8,label=f'$n_{{corr,R}}=1$ $n_{{corr,Y}}=1$',lw=2,color=tuple(colors[2]))
ax.plot(Xs,N22s,marker='o',alpha=0.8,label=f'$n_{{corr,R}}=2$ $n_{{corr,Y}}=2$',lw=2,color=tuple(colors[3]))

#twin1.scatter(Xs_NR,N00s_NR,marker='o',alpha=0.8,color='b',label='w/o Radial Diffusion')


sigma = 1.0/(np.array(N00s))

popt,pconv = curve_fit(func,Xs,N00s,absolute_sigma=False)

Xmin = Xs[0]
Xmax = Xs[-1]



#twin1.plot(np.linspace(Xmin,Xmax),func(np.linspace(Xmin,Xmax),*popt),label=f'$Y=N_{{AB}}(1{popt[0]:+.2f}x^{{-1}}{popt[1]:+.2f}x^{{-2}}$\n${popt[2]:+.2f}x^{{-3}})$',lw=3,alpha=1.0,color=tuple(colors[6]))
ax.axhline(y=calcN_AB(r1,r2,r3),color=tuple(colors[7]),ls='--',lw=2,label=r'$N_{AB}$')

#twin1.plot(Xs_NR,N1s_NR,marker='o',label='w/o Radial Diffusion',alpha=0.7,color='r',ls='-.')
#ax.tick_params(axis='y',colors=tuple(colors[7]))
ax.set_ylabel('Collection\nEfficiency',fontsize='large',fontweight='bold')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
#ax.set_ylim(0.39,0.7)
#ax.set_xlim(1,6)

ax.legend(loc='upper right')

sec_ax = ax.secondary_xaxis(-0.2,functions=(to_radius,to_axis_1))
sec_ax.set_xlabel(r'$r_1, \mu m$ ',fontsize='large',fontweight='bold')
ax.set_xlabel(r'$Sc^{\frac{1}{3}}Re^{\frac{1}{2}}$',fontsize='large',fontweight='bold')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.36),
          fancybox=True, shadow=True,ncol=2)


fig.text(0.05,0.89,'a)',fontsize=20)
fig.text(0.05,0.62,'b)',fontsize=20)
fig.text(0.05,0.35,'c)',fontsize=20)


fig.savefig(f'./analysis/Sc Correction.png',dpi=250,bbox_inches='tight')
fig.savefig(f'E:\OneDrive - Nexus365\Project PINN Hydrodynamic\Paper 15 Figures\Figure 5.png',dpi=250,bbox_inches='tight')
plt.close('all')





##################################################################################################################











fig,ax = plt.subplots(figsize=(16,9))
#twin1 = ax.twinx()

ax.plot(Xs,N00s,marker='o',alpha=0.8,color='k',label=f'$n_{{corr,R}}=0$ $n_{{corr,Y}}=0$',lw=2)
ax.plot(Xs,N01s,marker='o',alpha=0.8,label=f'$n_{{corr,R}}=0$ $n_{{corr,Y}}=1$',lw=2)
ax.plot(Xs,N10s,marker='o',alpha=0.8,label=f'$n_{{corr,R}}=1$ $n_{{corr,Y}}=0$',lw=2)

ax.plot(Xs,N11s,marker='*',alpha=0.8,label=f'$n_{{corr,R}}=1$ $n_{{corr,Y}}=1$',lw=2)
#twin1.scatter(Xs_NR,N00s_NR,marker='o',alpha=0.8,color='b',label='w/o Radial Diffusion')
ax.legend()
"""
coeffs = np.polyfit(Xs,N0s,deg=2)
p3 = np.poly1d(coeffs)
print(ID,coeffs)
twin1.plot(Xs,p3(Xs),label='w/ Radial Diffusion',lw=3,alpha=1.0,color=tuple(colors[2]))
"""
sigma = 1.0/(np.array(N00s))

popt,pconv = curve_fit(func,Xs,N00s,absolute_sigma=False)

Xmin = Xs[0]
Xmax = Xs[-1]



#twin1.plot(np.linspace(Xmin,Xmax),func(np.linspace(Xmin,Xmax),*popt),label=f'$Y=N_{{AB}}(1{popt[0]:+.2f}x^{{-1}}{popt[1]:+.2f}x^{{-2}}$\n${popt[2]:+.2f}x^{{-3}})$',lw=3,alpha=1.0,color=tuple(colors[6]))
ax.axhline(y=calcN_AB(r1,r2,r3),color=tuple(colors[7]),ls='--',lw=2,label=r'$N_{AB}$')

#twin1.plot(Xs_NR,N1s_NR,marker='o',label='w/o Radial Diffusion',alpha=0.7,color='r',ls='-.')
ax.tick_params(axis='y',colors=tuple(colors[7]))
ax.set_ylabel('Collection\nEfficiency',fontsize='large',fontweight='bold',color=tuple(colors[7]))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
#ax.set_ylim(0.39,0.7)
#ax.set_xlim(1,6)

ax.legend(loc='upper right')

sec_ax = ax.secondary_xaxis(-0.2,functions=(to_radius,to_axis_1))
sec_ax.set_xlabel(r'$r_1, \mu m$ ',fontsize='large',fontweight='bold')
ax.set_xlabel(r'$Sc^{\frac{1}{3}}Re^{\frac{1}{2}}$',fontsize='large',fontweight='bold')

fig.savefig(f'./analysis/Round 10 N01.png',dpi=250,bbox_inches='tight')
plt.close('all')








fig,ax = plt.subplots(figsize=(16,9))
#twin1 = ax.twinx()

ax.plot(Xs,N00s,marker='o',alpha=0.8,color='k',label=f'$n_{{corr,R}}=0$ $n_{{corr,Y}}=0$',lw=2)
ax.plot(Xs,N11s,marker='o',alpha=0.8,label=f'$n_{{corr,R}}=1$ $n_{{corr,Y}}=1$',lw=2)
ax.plot(Xs,N12s,marker='o',alpha=0.8,label=f'$n_{{corr,R}}=1$ $n_{{corr,Y}}=2$',lw=2)
ax.plot(Xs,N21s,marker='o',alpha=0.8,label=f'$n_{{corr,R}}=2$ $n_{{corr,Y}}=1$',lw=2)

#twin1.scatter(Xs_NR,N00s_NR,marker='o',alpha=0.8,color='b',label='w/o Radial Diffusion')
ax.legend()
"""
coeffs = np.polyfit(Xs,N0s,deg=2)
p3 = np.poly1d(coeffs)
print(ID,coeffs)
twin1.plot(Xs,p3(Xs),label='w/ Radial Diffusion',lw=3,alpha=1.0,color=tuple(colors[2]))
"""
sigma = 1.0/(np.array(N00s))

popt,pconv = curve_fit(func,Xs,N00s,absolute_sigma=False)

Xmin = Xs[0]
Xmax = Xs[-1]



#twin1.plot(np.linspace(Xmin,Xmax),func(np.linspace(Xmin,Xmax),*popt),label=f'$Y=N_{{AB}}(1{popt[0]:+.2f}x^{{-1}}{popt[1]:+.2f}x^{{-2}}$\n${popt[2]:+.2f}x^{{-3}})$',lw=3,alpha=1.0,color=tuple(colors[6]))
ax.axhline(y=calcN_AB(r1,r2,r3),color=tuple(colors[7]),ls='--',lw=2,label=r'$N_{AB}$')

#twin1.plot(Xs_NR,N1s_NR,marker='o',label='w/o Radial Diffusion',alpha=0.7,color='r',ls='-.')
ax.tick_params(axis='y',colors=tuple(colors[7]))
ax.set_ylabel('Collection\nEfficiency',fontsize='large',fontweight='bold',color=tuple(colors[7]))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
#ax.set_ylim(0.39,0.7)
#ax.set_xlim(1,6)

ax.legend(loc='upper right')

sec_ax = ax.secondary_xaxis(-0.2,functions=(to_radius,to_axis_1))
sec_ax.set_xlabel(r'$r_1, \mu m$ ',fontsize='large',fontweight='bold')
ax.set_xlabel(r'$Sc^{\frac{1}{3}}Re^{\frac{1}{2}}$',fontsize='large',fontweight='bold')
ax.set_xlabel(r'$Sc^{\frac{1}{3}}Re^{\frac{1}{2}}$',fontsize='large',fontweight='bold')
fig.savefig(f'./analysis/Round 10 N2.png',dpi=250,bbox_inches='tight')
plt.close('all')




