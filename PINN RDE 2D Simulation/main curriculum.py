import numpy as np 
import os
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tensorflow.keras import callbacks
from tensorflow.python.keras.callbacks import Callback
from lib.pinn import PINN
from lib.network import Network
import pandas as pd
import tensorflow as tf
from helper import ConcMontella
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7680)])
from tensorflow.keras.callbacks import LearningRateScheduler
import math


linewidth = 4
fontsize = 15

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs

no_flux_bnd = False

# build a core network model
network = Network.build()
network.summary()
# build a PINN model

if no_flux_bnd:
    pinn = PINN(network).build_no_flux()
else:
    pinn = PINN(network).build()

pinn.compile(optimizer='Adam',loss='mse')

default_weight_name = "./weights/default.h5"
# pinn.save_weights(default_weight_name)


X_s = []
J_sses = []
R0_s = []

def prediction(epochs=50,maxT=0,R0=0.5,re=1e-5,num_train_samples=int(2e6),initial_weights = None,train=True,saving_directory="./Data",alpha=1.0):
    """
    re: Radius of Electrode, m
    """


    # Dimensional Diffusion coefficient

    freq = 50 #freq of disc electrode 
    omega = freq * np.pi*2 # Rotational freq
    nu = 1e-6 #s^-1 kinematic viscosity
    L =  0.51023*omega**1.5*nu**(-0.5)


    D = 1e-9 # m^2 s^-1, diffusion coefficients 

    script_L = L * re**3 / D # The dimensionless form of L


    Sc = nu/D # Schmidt number
    Re = re**2*omega/nu 

    FluxConvertToHale = np.sqrt(1.65894)/(((L/D)**(1/3))*re)
    TimeConvertToHale = (L/D)**(2/3)*(re**2)
    SigmaConvertToHale = (D/L)**(2/3)/(re**2)
    delta = 1.2865*(D/L)**(1/3) # diffusion layer thickness, m 
    Delta = delta/re # dimensionless diffusion layer thickness


    def schedule(epoch,lr):
        # a learning rate scheduler 
        if epoch<=50:
            return lr 
        else:
            lr *= alpha
            return lr
        

    # saving directory is where data(voltammogram, concentration profile etc is saved)
    if not os.path.exists(saving_directory):
        os.mkdir(saving_directory)
    
    #weights folder is where weights is saved
    if not os.path.exists('./weights'):
        os.mkdir('./weights')

    # number of training samples
    num_train_samples = int(num_train_samples)
    # number of test samples
    num_test_samples = 1000

    batch_size = 250

    #R0 = max(0.5,1.0 - 3*Delta)


    print(R0)

    maxT = maxT
    maxY = Delta*4
    maxR = max(4,1+Delta*4)


    # prefix or suffix of files 
    file_name = f'R0={R0:.2E} re={re:.2E} maxY={maxY:.2E} maxR={maxR:.2E} alpha={alpha:.2f} no_flux = {no_flux_bnd} epochs={epochs:.2E} n_train={num_train_samples:.2E}'


    




    # solve the domain problem
    TYR_dmn0 = np.random.rand(num_train_samples,3)
    TYR_dmn0[:,0] = 0.0
    TYR_dmn0[:,1] = TYR_dmn0[:,1] * maxY
    TYR_dmn0[:,2] = TYR_dmn0[:,2] * (maxR-R0) + R0

    aux1_dmn0 = np.random.rand(num_train_samples,1)
    aux1_dmn0[:,0] = 1.0/TYR_dmn0[:,2]

    aux2_dmn0 = np.random.rand(num_train_samples,1) # scriptL * Y **2 
    aux2_dmn0[:,0] = script_L * TYR_dmn0[:,1]**2

    aux3_dmn0 = np.random.rand(num_train_samples,1) # scriptL * R * Y
    aux3_dmn0[:,0] = script_L * TYR_dmn0[:,2] * TYR_dmn0[:,1]


    TYR_bnd0 = np.random.rand(num_train_samples,3)
    TYR_bnd0[:,0] = 0.0
    TYR_bnd0[:,1] = 0.0
    TYR_bnd0[:,2] = TYR_bnd0[:,2] * (1.0-R0) + R0

    TYR_bnd1 = np.random.rand(num_train_samples,3)
    TYR_bnd1[:,0] = 0.0
    TYR_bnd1[:,1] = 0.0 
    TYR_bnd1[:,2] = TYR_bnd1[:,2] * (maxR-1.0) + 1.0 

    TYR_bnd2 = np.random.rand(num_train_samples,3)
    TYR_bnd2[:,0] = 0.0
    TYR_bnd2[:,1] = TYR_bnd2[:,1] * maxY
    TYR_bnd2[:,2] = maxR

    TYR_bnd3 = np.random.rand(num_train_samples,3)
    TYR_bnd3[:,0] = 0.0
    TYR_bnd3[:,1] = maxY
    TYR_bnd3[:,2] = (maxR-R0)*TYR_bnd3[:,2] + R0

    TYR_bnd4 = np.random.rand(num_train_samples,3)
    TYR_bnd4[:,0] = 0.0
    TYR_bnd4[:,1] = TYR_bnd4[:,1] * maxY
    TYR_bnd4[:,2] = R0


    C_dmn0 = np.zeros((num_train_samples,1))
    C_bnd0 = np.zeros((num_train_samples,1))
    C_bnd1 = np.zeros((num_train_samples,1))
    C_bnd2 = np.ones((num_train_samples,1))
    C_bnd3 = np.ones((num_train_samples,1))
    if no_flux_bnd:
        C_bnd4 = np.zeros((num_train_samples,1))
    else:    
        C_bnd4 = ConcMontella((L/D)**(1/3)*TYR_bnd4[:,1]*re)


    

    x_train = [TYR_dmn0,aux1_dmn0,aux2_dmn0,aux3_dmn0,TYR_bnd0,TYR_bnd1,TYR_bnd2,TYR_bnd3,TYR_bnd4]
    y_train = [C_dmn0,C_bnd0,C_bnd1,C_bnd2,C_bnd3,C_bnd4]

    


    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    # the loss weight of each loss componentan can be varied


    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
    pinn.load_weights(default_weight_name)
    # If train = True, fit the NN, else, use saved weights
    print(f'./weights/weights {file_name}.h5') 
    if train:
        if initial_weights:
            pinn.load_weights(initial_weights)
        pinn.fit(x=x_train,y=y_train,epochs=epochs,batch_size =batch_size,verbose=2,callbacks=[lr_scheduler_callback],shuffle=False)
        pinn.save_weights(f'./weights/weights {file_name}.h5')
    else:
        try:
            pinn.load_weights(f'./weights/weights {file_name}.h5')
        except:
            print('Weights does not exist\nStart training')
            if initial_weights:
                pinn.load_weights(initial_weights)
            pinn.fit(x=x_train,y=y_train,epochs=epochs,batch_size =batch_size,verbose=2,callbacks=[lr_scheduler_callback])
            pinn.save_weights(f'./weights/weights {file_name}.h5')
    

    
    # plot concentration profile in a certain time step
    time_sects = [0.0]
    for index,time_sect in enumerate(time_sects):
        XYR_test = np.zeros((int(num_test_samples**2),3))
        XYR_test[...,0] = time_sect
        Y_flat = np.linspace(0,maxY,num_test_samples)
        R_flat = np.linspace(R0,maxR,num_test_samples)
        Y,R = np.meshgrid(Y_flat,R_flat)
        XYR_test[...,1] = Y.flatten()
        XYR_test[...,2] = R.flatten()

        C = network.predict(XYR_test)
        C = C.reshape(Y.shape)

        fig,axes = plt.subplots(figsize=(8,9),nrows=2)
        plt.subplots_adjust(hspace=0.4)
        ax = axes[0]
        mesh = ax.pcolormesh(R,Y,C,shading='auto')

        with open(f'{saving_directory}/{file_name}.npy',mode='wb') as f:
            np.save(f,np.array([R,Y,C]))


        cbar = plt.colorbar(mesh,pad=0.05, aspect=10,ax=ax)
        cbar.set_label('C(Y,R)')
        cbar.mappable.set_clim(0, 1)
        
        ax.set_ylim(-maxY*0.05,maxY)
        ax.add_patch(Rectangle((R0,-maxY*0.05),(1.0-R0),maxY*0.05,edgecolor='k',facecolor='r'))
        ax.add_patch(Rectangle((1.0,-maxY*0.05),(maxR-1.0),maxY*0.05,edgecolor='k',facecolor='k'))


        R_flat = np.linspace(R0,maxR,num=500)
        R_i = R_flat[1] - R_flat[0]
        TYR_flux = np.zeros((len(R_flat),3))
        TYR_flux[:,2] = R_flat

        TYR_flux = tf.convert_to_tensor(TYR_flux,dtype=tf.float32)

        with tf.GradientTape() as g:
            g.watch(TYR_flux)
            C = network(TYR_flux)
        dC_dTYR = g.batch_jacobian(C, TYR_flux).numpy()
        Flux_density = dC_dTYR[...,1].reshape(-1)


        J_ss =  sum(Flux_density*R_i)
        # Convert to the flux using W coordinates
        J_ss = J_ss * FluxConvertToHale + R0

        ax.set_xlabel('R',fontsize='large',fontweight='bold')
        ax.set_ylabel('Y',fontsize='large',fontweight='bold')
        ax.set_title(r'$Sc^{\frac{1}{3}}Re^{\frac{1}{2}}$'+f'={Sc**(1/3)*Re**(1/2):.5f}')

        ax = axes[1]
        ax.plot(R_flat,Flux_density)
        with open(f'{saving_directory}/{file_name} flux density.npy',mode='wb') as f:
            np.save(f,np.array([R_flat,Flux_density]))

        ax.set_title(f'J={J_ss:.4f}')

        ax.set_ylabel('Flux Density',fontsize='large')
        ax.set_xlabel('R-axis',fontsize='large')




        fig.savefig(f'{saving_directory}/{file_name}.png',bbox_inches='tight')

        X_s.append(Sc**(1/3)*Re**(1/2))
        J_sses.append(J_ss)
        R0_s.append(R0)
    

    tf.keras.backend.clear_session()
    plt.close('all')


    return f'./weights/weights {file_name}.h5'





if __name__ =='__main__':
    
    maxT = 0.0 # maxT is zero as we want to solve a steady state problem independent of time

    epochs = 150

    saving_directory = f'Data No Flux = {no_flux_bnd}'
    initial_weights = default_weight_name
    for R0 in [0.05]:
        for num_train_samples in [3e6]:
            for re in np.arange(7e-6,4.3e-5,step=2e-6):
                initial_weights = prediction(epochs=epochs,R0=R0,re=re,alpha=0.98,num_train_samples=num_train_samples,train=False,initial_weights=initial_weights,saving_directory=saving_directory)
                df = pd.DataFrame({'X':X_s,"J_ss":J_sses,'R0':R0_s})
                df.to_csv(f'{saving_directory}/results.csv',index=False)


