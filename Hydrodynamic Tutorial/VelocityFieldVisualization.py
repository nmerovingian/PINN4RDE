import numpy as np 
import os
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
from matplotlib.colors import Normalize
colors = cm.tab10(np.linspace(0,1,9))



NO_FLUX_BND = False  # This is a boolean variable for no_flux or fixed boundary condition. The fixed boundary condition is a condition provided by Levich approximation. 
RADIALDIFFUSION=True



linewidth = 4
fontsize = 15

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs




def plot(Tsim=0,freq=1.0,R0=0.05,r1=1e-5,n_correction_terms = 2,saving_directory="./Figures"):
    """
    r1, Radius of Disk Electrode, m
    The length of the gap is simply r2-r1, m
    freq: Rotational Speed, in Hz
    R0: The dimensionless starting point of simulation in the R-direction
    r1: radius of disk electrode, m
    """

    if not os.path.exists(saving_directory):
        os.mkdir(saving_directory)

    # Dimensional Diffusion coefficient


    omega = freq * np.pi*2 # Rotational freq
    nu = 1e-6 #s^-1 kinematic viscosity
    L =  0.51023*omega**1.5*nu**(-0.5)


    D = 1e-9 # m^2 s^-1, diffusion coefficients 

    scriptL = L * r1**3 / D # The dimensionless form of L


    Sc = nu/D # Schmidt number
    Re = r1**2*omega/nu  # Reynold number

    delta = 1.2865*(D/L)**(1/3) # diffusion layer thickness, m 
    Delta = delta/r1 # dimensionless diffusion layer thickness


    R1 = r1/r1 # R1 is the dimnesionless radius of disk electrode, should always be 1. 

    Tsim = Tsim # MaxT is always zero since this problem is time independnet. 
    Ysim = Delta*4
    Rsim = 5


    fig,ax = plt.subplots(figsize=(8,4.5))

    # Velocity filed Visualization
    num_test_samples=int(2e3)
    TYR_dmn0 = np.zeros((int(num_test_samples**2),3))
    TYR_dmn0[...,0] = 0.0
    Y_flat = np.linspace(0,Ysim,num_test_samples)
    R_flat = np.linspace(R0,Rsim,num_test_samples)
    R,Y = np.meshgrid(R_flat,Y_flat)
    TYR_dmn0[...,1] = Y.flatten()
    TYR_dmn0[...,2] = R.flatten()


    auxY_dmn0 = np.random.rand(int(num_test_samples**2),1) # The convection velocity in the Y direction. 

    f1 = 0.33333/0.51023 * TYR_dmn0[:,1]**1*Re**(0.5)
    f2 = 0.10265/0.51023 * TYR_dmn0[:,1]**2*Re**(1.0)
    f3 = 0.01265/0.51023 * TYR_dmn0[:,1]**3*Re**(1.5)
    f4 = 0.00283/0.51023 * TYR_dmn0[:,1]**4*Re**(2)
    f5 = 0.00179/0.51023 * TYR_dmn0[:,1]**5*Re**(2.5)
    f6 = 0.00045/0.51023 * TYR_dmn0[:,1]**6*Re**(3)

    if n_correction_terms == 0: 
        auxY_dmn0[:,0] = - scriptL * (TYR_dmn0[:,1])**2
    elif n_correction_terms == 1:
        auxY_dmn0[:,0] = - scriptL * (TYR_dmn0[:,1])**2 * (1.0 - f1)
    elif n_correction_terms == 2:
        auxY_dmn0[:,0] = - scriptL * (TYR_dmn0[:,1])**2 * (1.0 - f1 + f2)
    elif n_correction_terms == 3:
        auxY_dmn0[:,0] = - scriptL * (TYR_dmn0[:,1])**2 * (1.0 - f1 + f2 - f3)
    elif n_correction_terms == 4:
        auxY_dmn0[:,0] = - scriptL * (TYR_dmn0[:,1])**2 * (1.0 - f1 + f2 - f3 - f4)
    elif n_correction_terms == 5:
        auxY_dmn0[:,0] = - scriptL * (TYR_dmn0[:,1])**2 * (1.0 - f1 + f2 - f3 - f4 + f5)
    elif n_correction_terms == 6:
        auxY_dmn0[:,0] = - scriptL * (TYR_dmn0[:,1])**2 * (1.0 - f1 + f2 - f3 - f4 + f5 - f6)
    else:
        raise ValueError 

    auxR_dmn0 =  np.random.rand(int(num_test_samples**2),1) # The convection velocity in the R direction. 

    g1 = 0.50000/0.51023 * TYR_dmn0[:,1]**1*Re**(0.5)
    g2 = 0.20533/0.51023 * TYR_dmn0[:,1]**2*Re**(1.0)
    g3 = 0.03060/0.51023 * TYR_dmn0[:,1]**3*Re**(1.5)
    g4 = 0.00850/0.51023 * TYR_dmn0[:,1]**4*Re**(2.0)
    g5 = 0.00628/0.51023 * TYR_dmn0[:,1]**5*Re**(2.5)
    g6 = 0.00180/0.51023 * TYR_dmn0[:,1]**6*Re**(3.0)

    if n_correction_terms == 0:
        auxR_dmn0[:,0] = scriptL * TYR_dmn0[:,2] * TYR_dmn0[:,1] * (1.0)
    elif n_correction_terms == 1:
        auxR_dmn0[:,0] = scriptL * TYR_dmn0[:,2] * TYR_dmn0[:,1] * (1.0 - g1)
    elif n_correction_terms == 2:
        auxR_dmn0[:,0] = scriptL * TYR_dmn0[:,2] * TYR_dmn0[:,1] * (1.0 - g1 + g2)
    elif n_correction_terms == 3:
        auxR_dmn0[:,0] = scriptL * TYR_dmn0[:,2] * TYR_dmn0[:,1] * (1.0 - g1 + g2 - g3)
    elif n_correction_terms == 4:
        auxR_dmn0[:,0] = scriptL * TYR_dmn0[:,2] * TYR_dmn0[:,1] * (1.0 - g1 + g2 - g3 - g4)
    elif n_correction_terms == 5:
        auxR_dmn0[:,0] = scriptL * TYR_dmn0[:,2] * TYR_dmn0[:,1] * (1.0 - g1 + g2 - g3 - g4 + g5)
    elif n_correction_terms == 6:
        auxR_dmn0[:,0] = scriptL * TYR_dmn0[:,2] * TYR_dmn0[:,1] * (1.0 - g1 + g2 - g3 - g4 + g5 - g6)
    else:
        raise ValueError

    auxY_dmn0 = auxY_dmn0.reshape(R.shape)
    auxR_dmn0 = auxR_dmn0.reshape(R.shape)


    print(R.shape,Y.shape, auxR_dmn0.shape, auxY_dmn0.shape)
    velocity_field = np.sqrt(auxR_dmn0**2+auxY_dmn0**2)
    ax.streamplot(R,Y,auxR_dmn0,auxY_dmn0,density=1.2,arrowsize=1.2,color=velocity_field,cmap='viridis')

    cbar = plt.colorbar(cm.ScalarMappable(norm=Normalize(vmin=np.min(velocity_field),vmax=np.max(velocity_field)),cmap='viridis'),ax=ax,pad=0.05,aspect=10)
    cbar.set_label(r'$||V(R,Y)||_2$')




    ax.set_ylim(-Ysim*0.05,Ysim)
    ax.add_patch(Rectangle((R0,-Ysim*0.05),(R1-R0),Ysim*0.05,edgecolor='k',facecolor='#FFD700'))
    ax.add_patch(Rectangle((R1,-Ysim*0.05),(Rsim-R1),Ysim*0.05,edgecolor='k',facecolor='k'))


    ax.set_xlabel('$R$',fontsize='large',fontweight='bold')
    ax.set_ylabel('$Y$',fontsize='large',fontweight='bold')


    ax.set_title(f'$r_e$={r1:.2E}m, f={freq:.0f}Hz',fontsize='large',fontweight='bold')




    fig.tight_layout()

    fig.savefig(f'Velocity Field re = {r1:.2E} f={freq:.0f}Hz.png',dpi=250,bbox_inches='tight')

if __name__ == "__main__":
    plot(freq=5,r1=1e-5)
    plot(freq=50,r1=1e-5)
    plot(freq=5,r1=1e-3)
    plot(freq=50,r1=1e-3)