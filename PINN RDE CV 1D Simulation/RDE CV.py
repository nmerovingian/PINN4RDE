import tensorflow as tf 
import numpy as np 
from lib.network import Network
from lib.pinn import PINN
from Helper import calcW
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


def main(epochs=100,sigma=40.0,train=True,saving_directory='./Data'):
    
    if not os.path.exists(saving_directory):
        os.mkdir(saving_directory)


    # number of training samples
    n_train_samples = 1000000
    # number of test samples
    n_test_samples = 500  
    
    batch_size = 250

    # Starting and reversing potential of simulation 
    theta_i = 10.0
    theta_v = -10.0
    maxT = 2.0*abs(theta_v-theta_i)/sigma  # total time of voltammetric scan, just one cycle

    maxY = 0.999 # The maximum Hale transformed coordinates for simulation


    TY_eqn = np.random.rand(n_train_samples, 2)
    TY_eqn[:,0] = TY_eqn[:,0] * maxT
    TY_eqn[:,1] = TY_eqn[:,1] * maxY


    prefactor_eqn = np.exp(-2.0/3.0*calcW(TY_eqn[:,1])**3)/1.65894


    TY_ini = np.random.rand(n_train_samples,2)
    TY_ini[:,0] = 0.0
    TY_ini[:,1] = TY_ini[:,1] * maxY


    TY_bnd0 = np.random.rand(n_train_samples, 2)
    TY_bnd0[:,0] = TY_bnd0[:,0] * maxT
    TY_bnd0[:,1] = 0.0  

    TY_bnd1 = np.random.rand(n_train_samples, 2)
    TY_bnd1[:,0] = TY_bnd1[:,0] * maxT
    TY_bnd1[:,1] = maxY


    C_eqn = np.zeros((n_train_samples, 1))
    C_bnd0 = np.ones((n_train_samples, 1))
    C_bnd1 = np.ones((n_train_samples, 1))
    C_ini = np.ones((n_train_samples, 1))

    for i in range(n_train_samples):
        if TY_bnd0[i,0] < maxT/2.0:
            C_bnd0[i] = 1.0/(1.0+np.exp(-(theta_i-sigma*TY_bnd0[i,0])))
        else:
            C_bnd0[i] = 1.0/(1.0+np.exp(-(theta_v+sigma*(TY_bnd0[i,0]-maxT/2.0))))
    



    x_train = [TY_eqn,prefactor_eqn,TY_bnd0,TY_bnd1,TY_ini]
    y_train = [C_eqn,C_bnd0,C_bnd1,C_ini]

    file_name = f'sigma={sigma:.2E} epochs={epochs:.2E} n_train={n_train_samples:.2E}'
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




    # predict C(T,Y) distribution
    T_flat = np.linspace(0, maxT, n_test_samples)
    E_flat = np.where(T_flat<maxT/2.0,theta_i-sigma*T_flat,theta_v+sigma*(T_flat-maxT/2.0))
    Y_flat = np.linspace(0, maxY, n_test_samples) 
    T, Y = np.meshgrid(T_flat, Y_flat)
    TY = np.stack([T.flatten(), Y.flatten()], axis=-1)
    C = network.predict(TY, batch_size=n_test_samples)
    C = C.reshape(T.shape)
    Y_i = Y_flat[1]
    flux = -(C[1,:] - C[0,:])/Y_i
    df = pd.DataFrame({'Potential':E_flat,'Flux':flux})
    df.to_csv(f'{saving_directory}/CV {file_name}.csv',index=False)
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.plot(E_flat,flux,label='PINN prediction')


    ax.set_xlabel(r'Potential, $\theta$')
    ax.set_ylabel('Flux,J')
    ax.legend()
    fig.savefig(f'{saving_directory}/CV {file_name}.png',bbox_inches='tight')


    fig,ax = plt.subplots(figsize=(8,4.5))

    axes = ax.pcolormesh(T,Y,C,cmap='viridis',shading='auto')
    ax.set_xlabel('T',fontsize='large',fontweight='bold')
    ax.set_ylabel('Y',fontsize='large',fontweight='bold')
    cbar = plt.colorbar(axes,pad=0.05, aspect=10,ax=ax)

    cbar.set_label('C(T,Y)',fontsize='large',fontweight='bold')


    with open(f'{saving_directory}/{file_name}.npy','wb') as f:
        np.save(f,np.array([T,Y,C]))


    fig.savefig(f'{saving_directory}/ConcProfile {file_name}.png')

    plt.close('all')



if __name__ == "__main__":

    kinematic_viscosity = 1e-6 # m^2 s^-1 
    freq = 500  # Freqnecy of rotation, Hz
    nu = 1e-3 #scan rate, V/s
    L = 0.51023*(2*np.pi*freq)**1.5*nu*(-0.5) # L, the auxilliary variable 
    D = 1e-9 # diffusion coefficient

    sigma = 96485*nu/(8.314*298)*(L**2*1e-9)**(-1/3) # Dimensionless Scan Rate


    sigmas = [0.1,1,2,5,10,15,20,30,40,50,60,70,80,90,100]
    for sigma in sigmas:
        main(epochs=150,sigma=sigma,train=False)
    