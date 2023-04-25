import tensorflow as tf 
import numpy as np 
from lib.network import Network
from lib.pinn import PINN
import matplotlib.pyplot as plt
import pandas as pd
import os
from Helper import calcY
import math
import argparse
from itertools import product
import sys 
SLRUM_ARRAY_TASK_ID = int(sys.argv[1])

nu_list = [1.4e-6,1.2e-6,1e-6,8e-7,5e-7,2e-7,1e-7]
sigma_list = [0.1,0.5]

result = product(nu_list,sigma_list)
result = list(result)

nu,sigma = result[SLRUM_ARRAY_TASK_ID]



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

default_weight_name = f"./weights/default {SLRUM_ARRAY_TASK_ID}.h5"

from random import randint
from time import sleep

sleep(randint(3,10))

pinn.save_weights(default_weight_name)




freq = 50 #Hz
omega = freq * np.pi*2 # rad/sec
#nu = 1e-7 #m^2 s^-1 kinematic viscosity
L =  0.51023*omega**1.5*nu**(-0.5)


re = 1e-5 # m, radius of electrode 
D = 1e-9 # m^2 s^-1, diffusion coefficients 
script_L = L * re**3 / D

Sc = nu/D


FluxConvertToHale = np.sqrt(1.65894)/(((L/D)**(1/3))*re)
TimeConvertToHale = (L/D)**(2/3)*(re**2)
SigmaConvertToHale = (D/L)**(2/3)/(re**2)
delta = 1.2865*(D/L)**(1/3) # diffusion layer thickness, m 
Delta = delta/re





Lambdas = []
J_sses = []
J_newman = []
scan_rates =[]
expansion_terms  = []



def main(epochs=100,sigma=40.0,Lambda_Ratio = 5,train=True,saving_directory='Results',expansion_term = 0,decay_rate=0.03):


    def schedule(epoch,lr):
        # a learning rate scheduler 
        if epoch<=50:
            return lr 
        else:
            lr *= (1.0-decay_rate)
            return lr


    # number of training samples
    n_train_samples = 1000000
    # number of test samples
    n_test_samples = 2000  

    batch_size = 250

    theta_i = 10.0
    theta_v = -10.0
    maxT = 2.0*abs(theta_v-theta_i)/sigma  # total time of voltammetric scan 

    
    #maxY = 2*np.sqrt(maxT)
    maxW = (L/D)**(1/3)*delta*Lambda_Ratio

    TW_eqn = np.random.rand(n_train_samples, 2)
    TW_eqn[:,0] = TW_eqn[:,0] * maxT
    TW_eqn[:,1] = TW_eqn[:,1] * maxW

    #f1 to f6 are six Schmidt number correction terms
    f1 = 0.33333/0.51023 * TW_eqn[:,1]**1/(0.51023**(1/3))*(Sc**(-1/3))
    f2 = 0.10265/0.51023 * TW_eqn[:,1]**2/(0.51023**(2/3)) *(Sc**(-2/3))
    f3 = 0.01265/0.51023 * TW_eqn[:,1]**3/(0.51023**(1))*(Sc**(-1))
    f4 = 0.00283/0.51023 * TW_eqn[:,1]**4/(0.51023**(4/3))*(Sc**(-4/3))
    f5 = 0.00179/0.51023 * TW_eqn[:,1]**5/(0.51023**(5/3))*(Sc**(-5/2))
    f6 = 0.00045/0.51023 * TW_eqn[:,1]**6/(0.51023**(2))*(Sc**(-2))

    if expansion_term == 0:
        prefactor_eqn = (TW_eqn[:,1])**2
    elif expansion_term == 1:
        prefactor_eqn = (TW_eqn[:,1])**2 *(1.0 - f1)
    elif expansion_term == 2:
        prefactor_eqn = (TW_eqn[:,1])**2 *(1.0-f1 + f2)
    elif expansion_term == 3:
        prefactor_eqn = (TW_eqn[:,1])**2 *(1.0-f1 + f2 - f3)
    elif expansion_term == 4:
        prefactor_eqn = (TW_eqn[:,1])**2 *(1.0-f1 + f2 - f3 - f4)
    elif expansion_term == 5:
        prefactor_eqn = (TW_eqn[:,1])**2 *(1.0-f1 + f2 - f3 - f4 + f5)
    elif expansion_term == 6:
        prefactor_eqn = (TW_eqn[:,1])**2 *(1.0-f1 + f2 - f3 - f4 + f5 - f6)



    TW_ini = np.random.rand(n_train_samples,2)
    TW_ini[:,0] = 0.0
    TW_ini[:,1] = TW_ini[:,1] * maxW


    TW_bnd0 = np.random.rand(n_train_samples, 2)
    TW_bnd0[:,0] = TW_bnd0[:,0] * maxT
    TW_bnd0[:,1] = 0.0  

    TW_bnd1 = np.random.rand(n_train_samples, 2)
    TW_bnd1[:,0] = TW_bnd1[:,0] * maxT
    TW_bnd1[:,1] = maxW


    C_eqn = np.zeros((n_train_samples, 1))
    C_bnd0 = np.ones((n_train_samples, 1))
    C_bnd1 = np.ones((n_train_samples, 1))
    C_ini = np.ones((n_train_samples, 1))

    for i in range(n_train_samples):
        if TW_bnd0[i,0] < maxT/2.0:
            C_bnd0[i] = 1.0/(1.0+np.exp(-(theta_i-sigma*TW_bnd0[i,0])))
        else:
            C_bnd0[i] = 1.0/(1.0+np.exp(-(theta_v+sigma*(TW_bnd0[i,0]-maxT/2.0))))
    


    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)

    x_train = [TW_eqn,prefactor_eqn,TW_bnd0,TW_bnd1,TW_ini]
    y_train = [C_eqn,C_bnd0,C_bnd1,C_ini]

    file_name = f'{saving_directory}/sigma={sigma:.2E} Sc= {Sc:.2E} epochs={epochs:.2E} Lambda_ratio={Lambda_Ratio:.1f} n_train={n_train_samples} expansion_terms = {expansion_term} dr={decay_rate:.2f}'
    weights_file_name = f'sigma={sigma:.2E} Sc= {Sc:.2E} epochs={epochs:.2E} Lambda_ratio={Lambda_Ratio:.1f} n_train={n_train_samples} expansion_terms = {expansion_term} dr={decay_rate:.2f}'
    #pinn.load_weights(default_weight_name)
    if train:
        pinn.fit(x=x_train,y=y_train,epochs=epochs,verbose=2,batch_size=batch_size,callbacks=[lr_scheduler_callback])
        pinn.save_weights(f'./weights/weights {weights_file_name}.h5')
    else:
        try:
            pinn.load_weights(f'./weights/weights {weights_file_name}.h5')
        except:
            print('Weights does not exist\nStart training')
            print(f'Weights name: ./weights/weights {weights_file_name}.h5')
            pinn.fit(x=x_train,y=y_train,epochs=epochs,verbose=2,batch_size=batch_size,callbacks=[lr_scheduler_callback])
            pinn.save_weights(f'./weights/weights {weights_file_name}.h5')




    # predict c(t,x) distribution
    T_flat = np.linspace(0, maxT, n_test_samples)
    E_flat = np.where(T_flat<maxT/2.0,theta_i-sigma*T_flat,theta_v+sigma*(T_flat-maxT/2.0))
    W_flat = np.linspace(0, maxW, n_test_samples) 
    T, W = np.meshgrid(T_flat, W_flat)
    TW = np.stack([T.flatten(), W.flatten()], axis=-1)
    C = network.predict(TW, batch_size=n_test_samples)
    C = C.reshape(T.shape)
    W_i = W_flat[1]
    Y_i = calcY(W_i)
    flux = -(C[1,:] - C[0,:])/Y_i



    # calculate steady state flux
    steady_state_flux = np.average(flux[int(len(flux)/2-10):int(len(flux)/2+10)])
    Newman_flux = -1.0/(1+0.2980*Sc**(-1/3)+0.14514*Sc**(-2/3))

    J_newman.append(Newman_flux)

    df = pd.DataFrame({'Potential':E_flat,'Flux':flux})
    df.to_csv(f'{file_name} CV.csv',index=False)
    fig, ax = plt.subplots(figsize=(16,9))
    ax.plot(E_flat,flux,label=f'PINN prediction, $\Lambda$={Lambda_Ratio:.1f}\n$J_{{ss}}={steady_state_flux:.4f}$ Expansion Terms={expansion_term}\nSc={Sc:.0f} J_newman = {Newman_flux:.4f}')

    


    ax.set_xlabel(r'Potential, $\theta$')
    ax.set_ylabel('Flux,J')
    ax.legend()
    fig.savefig(f'{file_name} CV.png',bbox_inches='tight',dpi=250)


    fig,ax = plt.subplots(figsize=(8,4.5))

    axes = ax.pcolormesh(T,W,C,cmap='viridis',shading='auto')
    with open(f'{file_name}.npy',mode='wb') as f:
        np.save(f,np.array([T,W,C]))
    ax.set_xlabel('T',fontsize='large',fontweight='bold')
    ax.set_ylabel('W',fontsize='large',fontweight='bold')
    ax.set_title(f'$\sigma={sigma:.2E}$ $\Lambda={Lambda_Ratio:.1f}$')
    cbar = plt.colorbar(axes,pad=0.05, aspect=10,ax=ax)

    cbar.set_label('C(T,W)',fontsize='large',fontweight='bold')




    fig.savefig(f'{file_name} ConcProfile.png',bbox_inches='tight',dpi=250)
    plt.close('all')

    return Lambda_ratio, steady_state_flux


if __name__ == "__main__":

    epochs = 100
    saving_directory = f'PINN sigma= {sigma:.2f} Sc={Sc:.0f}'
    if not os.path.exists(saving_directory):
        os.mkdir(saving_directory)


    for Lambda_ratio in [5]:
        for expansion_term in [0,1,2,3,4,5,6]:

            Lambda_ratio, J_ss=main(epochs=epochs,sigma=sigma,Lambda_Ratio=Lambda_ratio,train=False,saving_directory=saving_directory,expansion_term=expansion_term)
            scan_rates.append(sigma)
            Lambdas.append(Lambda_ratio)
            J_sses.append(J_ss)
            expansion_terms.append(expansion_term)

            df = pd.DataFrame({'scan rate':scan_rates,"Lambda":Lambdas,"Expansion Term":expansion_terms,"J_ss":J_sses,"J_newman":J_newman})
            df['Pct Dif From Levich'] = (df['J_ss'] - (-1))/df['J_ss']
            df['PCT Dif From Newman'] = (df['J_ss'] - df['J_newman'])/df['J_newman']

            df.to_csv(f'{saving_directory}/Results.csv',index=False)