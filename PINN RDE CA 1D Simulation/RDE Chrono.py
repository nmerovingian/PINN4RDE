import tensorflow as tf 
import numpy as np 
from lib.network import Network
from lib.pinn import PINN

import matplotlib.pyplot as plt
import pandas as pd
import os

linewidth = 4
fontsize = 20

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs


#weights folder is where weights is saved
if not os.path.exists('./weights'):
    os.mkdir('./weights')


network = Network.build()
network.summary()
# build a PINN model
pinn = PINN(network).build()
pinn.compile(optimizer='adam',loss='mse')

default_weight_name = "./weights/default.h5"
pinn.save_weights(default_weight_name)


freq = 50 #Hz
Omega = 50 * np.pi*2 # rad/sec
nu = 1e-6 #m^2 s^-1 kinematic viscosity
A_factor =  0.51023*Omega**1.5*nu**-0.5 
re = 1e-5 # m, radius of electrode 
D = 1e-9 # m^2 s^-1, diffusion coefficients 
script_A_factor = A_factor * re**3 / D
FluxConvertToHale = np.sqrt(1.65894)/(((A_factor/D)**(1/3))*re)
TimeConvertToHale = (A_factor/D)**(2/3)*(re**2)

print(TimeConvertToHale)
def main(epochs=100,maxT=1.0,train=True):

    # number of training samples
    n_train_samples = 1000000
    # number of test samples
    n_test_samples = 500  

    batch_size = 250
    stallT = 0.1
    maxY = 2*np.sqrt(maxT)


    TY_eqn = np.random.rand(n_train_samples, 2)
    TY_eqn[:,0] = TY_eqn[:,0] * maxT
    TY_eqn[:,1] = TY_eqn[:,1] * maxY


    prefactor_eqn = script_A_factor*(TY_eqn[:,1])**2


    TY_ini = np.random.rand(n_train_samples,2)
    TY_ini[:,0] =  0.0
    TY_ini[:,1] = TY_ini[:,1] * maxY


    TY_bnd0 = np.random.rand(n_train_samples, 2)
    TY_bnd0[:,0] = TY_bnd0[:,0] * maxT
    TY_bnd0[:,1] = 0.0  

    TY_bnd1 = np.random.rand(n_train_samples, 2)
    TY_bnd1[:,0] = TY_bnd1[:,0] * maxT
    TY_bnd1[:,1] =  maxY


    C_eqn = np.zeros((n_train_samples, 1))
    C_bnd0 = np.ones((n_train_samples, 1))
    C_bnd1 = np.ones((n_train_samples, 1))
    C_ini = np.ones((n_train_samples, 1))

    for i in range(n_train_samples):
        if TY_bnd0[i,0] > stallT:
            C_bnd0[i] = 0.0
    



    x_train = [TY_eqn,prefactor_eqn,TY_bnd0,TY_bnd1,TY_ini]
    y_train = [C_eqn,C_bnd0,C_bnd1,C_ini]


    file_name = f'maxT={maxT:.2E} epochs={epochs:.2E} n_train={n_train_samples}'
    pinn.load_weights(default_weight_name)
    if train:
        pinn.fit(x=x_train,y=y_train,epochs=epochs,verbose=2,batch_size=batch_size)
        pinn.save_weights(f'./weights/weights {file_name}.h5')
    else:
        try:
            pinn.load_weights(f'./weights/weights {file_name}.h5')
        except:
            print('Weights does not exist\nStart training')
            pinn.fit(x=x_train,y=y_train,epochs=epochs,verbose=1,batch_size=batch_size)
            pinn.save_weights(f'./weights/weights {file_name}.h5')




    # predict c(t,x) distribution
    T_flat = np.linspace(0, maxT, n_test_samples)

    Y_flat = np.linspace(0, maxY, n_test_samples) 
    T, Y = np.meshgrid(T_flat, Y_flat)
    TY = np.stack([T.flatten(), Y.flatten()], axis=-1)
    C = network.predict(TY, batch_size=n_test_samples)
    C = C.reshape(T.shape)
    Y_i = Y_flat[1]
    flux = -(C[1,:] - C[0,:])/Y_i
    # ConvertFlux to Flux after HaleTransformation 
    flux *= FluxConvertToHale
    T_flat *= TimeConvertToHale
    df = pd.DataFrame({'Time':T_flat,'Flux':flux})
    df.to_csv(f'chronoamperogram {file_name}.csv',index=False)
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.plot(T_flat,flux,label='PINN prediction')
    ax.axhline(y=-1,label=r'$J_{ss}$',color='k',ls='--')

    df = pd.read_csv('Bruckenstein & Prager.csv',header=None)
    df.iloc[:,1] = -df.iloc[:,1]
    df.iloc[:,0] += (stallT*TimeConvertToHale)
    ax.scatter(df.iloc[:,0],df.iloc[:,1],label='Bruckenstein & Prager',color='orange')




    ax.set_xlabel('Time, T')
    ax.set_ylabel('Flux,J')
    ax.legend()
    fig.savefig(f'Chronoamperogram {file_name}.png',bbox_inches='tight')


    fig,ax = plt.subplots(figsize=(8,4.5))

    axes = ax.pcolormesh(T,Y,C,cmap='viridis',shading='auto')
    ax.set_xlabel('T',fontsize='large',fontweight='bold')
    ax.set_ylabel('Y',fontsize='large',fontweight='bold')
    cbar = plt.colorbar(axes,pad=0.05, aspect=10,ax=ax)

    cbar.set_label('C(T,Y)',fontsize='large',fontweight='bold')


    fig.savefig(f'ConcProfile {file_name}.png',bbox_inches='tight')



if __name__ == "__main__":
    main(epochs=40,maxT=1,train=False)