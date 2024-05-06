import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tensorflow.keras import callbacks
from tensorflow.python.keras.callbacks import Callback
from tensorflow.keras.callbacks import LearningRateScheduler
import math
from lib.pinn import PINN
from lib.network import Network
import pandas as pd
import tensorflow as tf
from helper import ConcMontella
from NAnalytical import calcN_AB, calcN_IL
from itertools import product
import sys

results_directory = './results'  # Create this empty folder before programm running

# os.cpu_count() method
cpuCount = os.cpu_count()

# Print the number of 
# CPUs in the system 
print("Number of CPUs in the system:", cpuCount)

linewidth = 4
fontsize = 15

font = {'family': 'monospace',
        'weight': 'bold',
        'size': fontsize}
plt.rc('font', **font)  # pass in the font dict as kwargs

NO_FLUX_BND = False  # This is a boolean variable for no_flux or fixed boundary condition. The fixed boundary condition is a condition provided by Levich approximation. 
RADIALDIFFUSION = True

# build a core network model
network = Network.build()
# build a PINN model
pinn = PINN(network).build(NO_FLUX_BND, RADIALDIFFUSION)
pinn.compile(optimizer='Adam', loss='mse')

default_weight_name = f"./weights/default {TASK_ID}.h5"

from random import randint
from time import sleep

sleep(randint(1, 20))

if not os.path.exists(results_directory):
    os.mkdir(results_directory)

pinn.save_weights(default_weight_name)

r1s = []
r2s = []
r3s = []
freqs = []

X_s = []
J_Ds = []
J_Rs = []
J_D_Levichs = []
n_corrections_list_R = []
n_corrections_list_Y = []
Ns = []
NABs = []
NILs = []


class DataGenerator(tf.keras.utils.Sequence):
    # The data generator to generate training data on the fly.It save memory and may potentially be faster to train! 
    def __init__(self, r1, D, R0, R1, R2, R3, Re, Sc, L, scriptL, n_correction_R, n_correction_Y, Tsim, Rsim, Ysim,
                 num_train_samples, batch_size, ):
        super().__init__()
        self.r1 = r1  # The dimensional disk electrode radiusm, also used for dimension conversion
        self.D = D  # The dimensional diffusion coefficient
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.Re = Re  # Reynold Number 
        self.Sc = Sc  # Schdmit Number
        self.L = L
        self.scriptL = scriptL
        self.n_correction_R = n_correction_R  # Number of Sc number correction terms in the R direction
        self.n_correction_Y = n_correction_Y  # Number of Sc number correction terms in the Y direction
        self.Tsim = Tsim
        self.Rsim = Rsim
        self.Ysim = Ysim
        self.Ysim2 = Ysim * 0.4
        self.Rsim2 = R3 + 0.5
        self.num_train_samples = num_train_samples
        self.batch_size = batch_size

    def __getitem__(self, index):
        # Create training input
        # The convective diffusion equation with radial 
        TYR_dmn0 = np.random.rand(self.batch_size, 3)
        TYR_dmn0[:, 0] = 0.0
        TYR_dmn0[:, 1] = TYR_dmn0[:, 1] * self.Ysim
        TYR_dmn0[:, 2] = TYR_dmn0[:, 2] * (self.Rsim - self.R0) + self.R0

        # The 1/R is calculated here as:
        aux1overR_dmn0 = np.random.rand(self.batch_size, 1)
        aux1overR_dmn0[:, 0] = 1.0 / TYR_dmn0[:, 2]

        auxY_dmn0 = np.random.rand(self.batch_size, 1)  # The convection velocity in the Y direction.

        f1 = 0.33333 / 0.51023 * TYR_dmn0[:, 1] ** 1 * self.Re ** (0.5)
        f2 = 0.10265 / 0.51023 * TYR_dmn0[:, 1] ** 2 * self.Re ** (1.0)
        f3 = 0.01265 / 0.51023 * TYR_dmn0[:, 1] ** 3 * self.Re ** (1.5)
        f4 = 0.00283 / 0.51023 * TYR_dmn0[:, 1] ** 4 * self.Re ** (2)
        f5 = 0.00179 / 0.51023 * TYR_dmn0[:, 1] ** 5 * self.Re ** (2.5)
        f6 = 0.00045 / 0.51023 * TYR_dmn0[:, 1] ** 6 * self.Re ** (3)

        if self.n_correction_Y == 0:
            auxY_dmn0[:, 0] = - self.scriptL * (TYR_dmn0[:, 1]) ** 2
        elif self.n_correction_Y == 1:
            auxY_dmn0[:, 0] = - self.scriptL * (TYR_dmn0[:, 1]) ** 2 * (1.0 - f1)
        elif self.n_correction_Y == 2:
            auxY_dmn0[:, 0] = - self.scriptL * (TYR_dmn0[:, 1]) ** 2 * (1.0 - f1 + f2)
        elif self.n_correction_Y == 3:
            auxY_dmn0[:, 0] = - self.scriptL * (TYR_dmn0[:, 1]) ** 2 * (1.0 - f1 + f2 - f3)
        elif self.n_correction_Y == 4:
            auxY_dmn0[:, 0] = - self.scriptL * (TYR_dmn0[:, 1]) ** 2 * (1.0 - f1 + f2 - f3 - f4)
        elif self.n_correction_Y == 5:
            auxY_dmn0[:, 0] = - self.scriptL * (TYR_dmn0[:, 1]) ** 2 * (1.0 - f1 + f2 - f3 - f4 + f5)
        elif self.n_correction_Y == 6:
            auxY_dmn0[:, 0] = - self.scriptL * (TYR_dmn0[:, 1]) ** 2 * (1.0 - f1 + f2 - f3 - f4 + f5 - f6)
        else:
            raise ValueError

        auxR_dmn0 = np.random.rand(self.batch_size, 1)  # The convection velocity in the R direction.

        g1 = 0.50000 / 0.51023 * TYR_dmn0[:, 1] ** 1 * self.Re ** (0.5)
        g2 = 0.20533 / 0.51023 * TYR_dmn0[:, 1] ** 2 * self.Re ** (1.0)
        g3 = 0.03060 / 0.51023 * TYR_dmn0[:, 1] ** 3 * self.Re ** (1.5)
        g4 = 0.00850 / 0.51023 * TYR_dmn0[:, 1] ** 4 * self.Re ** (2.0)
        g5 = 0.00628 / 0.51023 * TYR_dmn0[:, 1] ** 5 * self.Re ** (2.5)
        g6 = 0.00180 / 0.51023 * TYR_dmn0[:, 1] ** 6 * self.Re ** (3.0)

        if self.n_correction_R == 0:
            auxR_dmn0[:, 0] = self.scriptL * TYR_dmn0[:, 2] * TYR_dmn0[:, 1] * (1.0)
        elif self.n_correction_R == 1:
            auxR_dmn0[:, 0] = self.scriptL * TYR_dmn0[:, 2] * TYR_dmn0[:, 1] * (1.0 - g1)
        elif self.n_correction_R == 2:
            auxR_dmn0[:, 0] = self.scriptL * TYR_dmn0[:, 2] * TYR_dmn0[:, 1] * (1.0 - g1 + g2)
        elif self.n_correction_R == 3:
            auxR_dmn0[:, 0] = self.scriptL * TYR_dmn0[:, 2] * TYR_dmn0[:, 1] * (1.0 - g1 + g2 - g3)
        elif self.n_correction_R == 4:
            auxR_dmn0[:, 0] = self.scriptL * TYR_dmn0[:, 2] * TYR_dmn0[:, 1] * (1.0 - g1 + g2 - g3 - g4)
        elif self.n_correction_R == 5:
            auxR_dmn0[:, 0] = self.scriptL * TYR_dmn0[:, 2] * TYR_dmn0[:, 1] * (1.0 - g1 + g2 - g3 - g4 + g5)
        elif self.n_correction_R == 6:
            auxR_dmn0[:, 0] = self.scriptL * TYR_dmn0[:, 2] * TYR_dmn0[:, 1] * (1.0 - g1 + g2 - g3 - g4 + g5 - g6)
        else:
            raise ValueError

        # The convective diffusion equation with radial
        TYR_dmn1 = np.random.rand(self.batch_size, 3)
        TYR_dmn1[:, 0] = 0.0
        TYR_dmn1[:, 1] = TYR_dmn1[:, 1] * self.Ysim2
        TYR_dmn1[:, 2] = TYR_dmn1[:, 2] * (self.Rsim2 - self.R0) + self.R0

        # The 1/R is calculated here as:
        aux1overR_dmn1 = np.random.rand(self.batch_size, 1)
        aux1overR_dmn1[:, 0] = 1.0 / TYR_dmn1[:, 2]

        auxY_dmn1 = np.random.rand(self.batch_size, 1)  # The convection velocity in the Y direction.

        f1 = 0.33333 / 0.51023 * TYR_dmn1[:, 1] ** 1 * self.Re ** (0.5)
        f2 = 0.10265 / 0.51023 * TYR_dmn1[:, 1] ** 2 * self.Re ** (1.0)
        f3 = 0.01265 / 0.51023 * TYR_dmn1[:, 1] ** 3 * self.Re ** (1.5)
        f4 = 0.00283 / 0.51023 * TYR_dmn1[:, 1] ** 4 * self.Re ** (2)
        f5 = 0.00179 / 0.51023 * TYR_dmn1[:, 1] ** 5 * self.Re ** (2.5)
        f6 = 0.00045 / 0.51023 * TYR_dmn1[:, 1] ** 6 * self.Re ** (3)

        if self.n_correction_Y == 0:
            auxY_dmn1[:, 0] = - self.scriptL * (TYR_dmn1[:, 1]) ** 2
        elif self.n_correction_Y == 1:
            auxY_dmn1[:, 0] = - self.scriptL * (TYR_dmn1[:, 1]) ** 2 * (1.0 - f1)
        elif self.n_correction_Y == 2:
            auxY_dmn1[:, 0] = - self.scriptL * (TYR_dmn1[:, 1]) ** 2 * (1.0 - f1 + f2)
        elif self.n_correction_Y == 3:
            auxY_dmn1[:, 0] = - self.scriptL * (TYR_dmn1[:, 1]) ** 2 * (1.0 - f1 + f2 - f3)
        elif self.n_correction_Y == 4:
            auxY_dmn1[:, 0] = - self.scriptL * (TYR_dmn1[:, 1]) ** 2 * (1.0 - f1 + f2 - f3 - f4)
        elif self.n_correction_Y == 5:
            auxY_dmn1[:, 0] = - self.scriptL * (TYR_dmn1[:, 1]) ** 2 * (1.0 - f1 + f2 - f3 - f4 + f5)
        elif self.n_correction_Y == 6:
            auxY_dmn1[:, 0] = - self.scriptL * (TYR_dmn1[:, 1]) ** 2 * (1.0 - f1 + f2 - f3 - f4 + f5 - f6)
        else:
            raise ValueError

        auxR_dmn1 = np.random.rand(self.batch_size, 1)  # The convection velocity in the R direction.

        g1 = 0.50000 / 0.51023 * TYR_dmn1[:, 1] ** 1 * self.Re ** (0.5)
        g2 = 0.20533 / 0.51023 * TYR_dmn1[:, 1] ** 2 * self.Re ** (1.0)
        g3 = 0.03060 / 0.51023 * TYR_dmn1[:, 1] ** 3 * self.Re ** (1.5)
        g4 = 0.00850 / 0.51023 * TYR_dmn1[:, 1] ** 4 * self.Re ** (2.0)
        g5 = 0.00628 / 0.51023 * TYR_dmn1[:, 1] ** 5 * self.Re ** (2.5)
        g6 = 0.00180 / 0.51023 * TYR_dmn1[:, 1] ** 6 * self.Re ** (3.0)

        if self.n_correction_R == 0:
            auxR_dmn1[:, 0] = self.scriptL * TYR_dmn1[:, 2] * TYR_dmn1[:, 1] * (1.0)
        elif self.n_correction_R == 1:
            auxR_dmn1[:, 0] = self.scriptL * TYR_dmn1[:, 2] * TYR_dmn1[:, 1] * (1.0 - g1)
        elif self.n_correction_R == 2:
            auxR_dmn1[:, 0] = self.scriptL * TYR_dmn1[:, 2] * TYR_dmn1[:, 1] * (1.0 - g1 + g2)
        elif self.n_correction_R == 3:
            auxR_dmn1[:, 0] = self.scriptL * TYR_dmn1[:, 2] * TYR_dmn1[:, 1] * (1.0 - g1 + g2 - g3)
        elif self.n_correction_R == 4:
            auxR_dmn1[:, 0] = self.scriptL * TYR_dmn1[:, 2] * TYR_dmn1[:, 1] * (1.0 - g1 + g2 - g3 - g4)
        elif self.n_correction_R == 5:
            auxR_dmn1[:, 0] = self.scriptL * TYR_dmn1[:, 2] * TYR_dmn1[:, 1] * (1.0 - g1 + g2 - g3 - g4 + g5)
        elif self.n_correction_R == 6:
            auxR_dmn1[:, 0] = self.scriptL * TYR_dmn1[:, 2] * TYR_dmn1[:, 1] * (1.0 - g1 + g2 - g3 - g4 + g5 - g6)
        else:
            raise ValueError

        # Dsik Electrode Surface
        TYR_bnd0 = np.random.rand(self.batch_size, 3)
        TYR_bnd0[:, 0] = 0.0
        TYR_bnd0[:, 1] = 0.0
        TYR_bnd0[:, 2] = TYR_bnd0[:, 2] * (self.R1 - self.R0) + self.R0
        # Gap surface 
        TYR_bnd1 = np.random.rand(self.batch_size, 3)
        TYR_bnd1[:, 0] = 0.0
        TYR_bnd1[:, 1] = 0.0
        TYR_bnd1[:, 2] = TYR_bnd1[:, 2] * (self.R2 - self.R1) + self.R1
        #Ring surface
        TYR_bnd2 = np.random.rand(self.batch_size, 3)
        TYR_bnd2[:, 0] = 0.0
        TYR_bnd2[:, 1] = 0.0
        TYR_bnd2[:, 2] = TYR_bnd2[:, 2] * (self.R3 - self.R2) + self.R2
        # Outer insulating surface, no flux boundary condition
        TYR_bnd3 = np.random.rand(self.batch_size, 3)
        TYR_bnd3[:, 0] = 0.0
        TYR_bnd3[:, 1] = 0.0
        TYR_bnd3[:, 2] = TYR_bnd3[:, 2] * (self.Rsim - self.R3) + self.R3

        # Left boudnary, no flux or Levich concentration profile apprximation
        TYR_bnd4 = np.random.rand(self.batch_size, 3)
        TYR_bnd4[:, 0] = 0.0
        TYR_bnd4[:, 1] = TYR_bnd4[:, 1] * self.Ysim
        TYR_bnd4[:, 2] = self.R0

        # Top boundary, fixed concentration boundary condition
        TYR_bnd5 = np.random.rand(self.batch_size, 3)
        TYR_bnd5[:, 0] = 0.0
        TYR_bnd5[:, 1] = self.Ysim
        TYR_bnd5[:, 2] = TYR_bnd5[:, 2] * (self.Rsim - self.R0) + self.R0

        # Right boundary, fxied concentration boundary condition
        TYR_bnd6 = np.random.rand(self.batch_size, 3)
        TYR_bnd6[:, 0] = 0.0
        TYR_bnd6[:, 1] = TYR_bnd6[:, 1] * self.Ysim
        TYR_bnd6[:, 2] = self.Rsim

        C_dmn0 = np.zeros((self.batch_size, 1))  # The residual of governing PDE should 0.
        C_dmn1 = np.zeros((self.batch_size, 1))  # The residual of governing PDE should 0.
        C_bnd0 = np.zeros((self.batch_size, 1))  # Fixed concentration of 0 at electrode surface
        C_bnd1 = np.zeros((self.batch_size, 1))  # No flux at the gap
        C_bnd2 = np.ones((self.batch_size, 1))  # Fixed concentraton of 1 at ring electrode
        C_bnd3 = np.zeros((self.batch_size, 1))  # No Flux bnd at outer insulating surface
        if NO_FLUX_BND:
            C_bnd4 = np.zeros((self.batch_size, 1))
        else:
            # The expression (L/D)**(1/3)*TYR_bnd4[:,1]*r1 is actually the W coordinate 
            C_bnd4 = ConcMontella((self.L / self.D) ** (1 / 3) * TYR_bnd4[:, 1] * self.r1)
        C_bnd5 = np.ones((self.batch_size, 1))  # Fixed bnd at top boundary
        C_bnd6 = np.ones((self.batch_size, 1))  # Fixed bnd at the right boundary

        x_train = [TYR_dmn0, aux1overR_dmn0, auxY_dmn0, auxR_dmn0, TYR_dmn1, aux1overR_dmn1, auxY_dmn1, auxR_dmn1,
                   TYR_bnd0, TYR_bnd1, TYR_bnd2, TYR_bnd3, TYR_bnd4, TYR_bnd5, TYR_bnd6]
        y_train = [C_dmn0, C_dmn1, C_bnd0, C_bnd1, C_bnd2, C_bnd3, C_bnd4, C_bnd5, C_bnd6]

        return x_train, y_train

    def __len__(self):
        return int(np.floor(self.num_train_samples / self.batch_size))


def prediction(epochs=50, Tsim=0, freq=5, R0=0.5, r1=1e-5, r2=2e-5, r3=3e-5, lambda_ratio=3.0, n_correction_R=0,
               n_correction_Y=0, num_train_samples=int(2e6), batch_size=250, initial_weights=None, train=True,
               saving_directory="./Data", alpha=1.0):
    """
    r1, Radius of Disk Electrode, m
    r2, Inner Radius of the Ring Electrode, m
    r3, Outer Radius of the Ring Electrode, m
    The length of the gap is simply r2-r1, m
    alpha, the learning rate decay factor, if alpha=1, no learning rate decay, if alpha<1, learning rate will decay. 
    freq  #freq of rotation,Hz
    """

    # Dimensional Diffusion coefficient

    omega = freq * np.pi * 2  # Rotational freq
    nu = 1e-6  #s^-1 kinematic viscosity
    L = 0.51023 * omega ** 1.5 * nu ** (-0.5)

    D = 1e-9  # m^2 s^-1, diffusion coefficients

    scriptL = L * r1 ** 3 / D  # The dimensionless form of L

    Sc = nu / D  # Schmidt number
    Re = r1 ** 2 * omega / nu  # Reynold number

    FluxConvertToHale = np.sqrt(1.65894) / (((L / D) ** (1 / 3)) * r1)  # Flux converted using Hale Transformation
    TimeConvertToHale = (L / D) ** (2 / 3) * (r1 ** 2)
    SigmaConvertToHale = (D / L) ** (2 / 3) / (r1 ** 2)
    delta = 1.2865 * (D / L) ** (1 / 3)  # diffusion layer thickness, m
    Delta = delta / r1  # dimensionless diffusion layer thickness
    deltaH = np.sqrt(nu / omega)  #hydrodynamic layer thickness
    DeltaH = deltaH / r1  #Dimensionless hydrodynamic layer thickness

    def schedule(epoch, lr):
        # a learning rate scheduler 
        if epoch <= 50:
            return 1e-3
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

    R1 = r1 / r1  # R1 is the dimnesionless radius of disk electrode, should always be 1.
    R2 = r2 / r1
    R3 = r3 / r1
    Tsim = Tsim  # MaxT is always zero since this problem is time independnet (steady state).
    #Ysim = Delta*(lambda_ratio+0.5)
    #Rsim = 4.0

    Ysim = max(3.0, Delta * (lambda_ratio + 0.5))
    Rsim = max(4.01, DeltaH)

    # prefix or suffix of files
    file_name = f'R0={R0:.2E} r1={r1:.2E} r2={r2:.2E} r3={r3:.2E} n_correction_R={n_correction_R} n_correction_Y={n_correction_Y} lambda_ratio = {lambda_ratio:.2E} maxY={Ysim:.2E} maxR={Rsim:.2E} alpha={alpha:.2f} epochs={epochs:.2E} No Flux = {NO_FLUX_BND} RadialD = {RADIALDIFFUSION}'

    training_generator = DataGenerator(r1=r1, D=D, R0=R0, R1=R1, R2=R2, R3=R3, Re=Re, Sc=Sc, L=L, scriptL=scriptL,
                                       n_correction_R=n_correction_R, n_correction_Y=n_correction_Y, Tsim=Tsim,
                                       Rsim=Rsim, Ysim=Ysim, num_train_samples=num_train_samples, batch_size=batch_size)

    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    # the loss weight of each loss componentan can be varied

    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
    pinn.load_weights(default_weight_name)
    # If train = True, fit the NN, else, use saved weights
    print(f'./weights/weights {file_name}.h5')
    if train:
        if initial_weights:
            pinn.load_weights(initial_weights)
        history = pinn.fit(training_generator, epochs=epochs, verbose=2, callbacks=[lr_scheduler_callback],
                           max_queue_size=10, workers=1, use_multiprocessing=False)
        history = pd.DataFrame(history.history)
        history.to_csv(f'{saving_directory}/ history {file_name}.csv')

        pinn.save_weights(f'./weights/weights {file_name}.h5')
    else:
        try:
            pinn.load_weights(f'./weights/weights {file_name}.h5')
        except:
            print('Weights does not exist\nStart training')
            if initial_weights:
                pinn.load_weights(initial_weights)
            history = pinn.fit(training_generator, epochs=epochs, verbose=2, callbacks=[lr_scheduler_callback],
                               max_queue_size=10, workers=1, use_multiprocessing=False)
            history = pd.DataFrame(history.history)
            history.to_csv(f'{saving_directory}/ history {file_name}.csv')
            pinn.save_weights(f'./weights/weights {file_name}.h5')

    # plot concentration profile in a certain time step
    time_sects = [0.0]
    for index, time_sect in enumerate(time_sects):
        XYR_test = np.zeros((int(num_test_samples ** 2), 3))
        XYR_test[..., 0] = time_sect
        Y_flat = np.linspace(0, Ysim, num_test_samples)
        R_flat = np.linspace(R0, Rsim, num_test_samples)
        Y, R = np.meshgrid(Y_flat, R_flat)
        XYR_test[..., 1] = Y.flatten()
        XYR_test[..., 2] = R.flatten()

        C = network.predict(XYR_test, verbose=2, batch_size=batch_size)
        C = C.reshape(Y.shape)

        fig, axes = plt.subplots(figsize=(8, 9), nrows=2)
        plt.subplots_adjust(hspace=0.4)
        ax = axes[0]
        mesh = ax.pcolormesh(R, Y, C, shading='auto')
        """
        with open(f'{saving_directory}/{file_name}.npy',mode='wb') as f:
            np.save(f,np.array([R,Y,C]))
        """
        cbar = plt.colorbar(mesh, pad=0.05, aspect=10, ax=ax)
        cbar.set_label('C(R,Y)')
        cbar.mappable.set_clim(0, 1)

        ax.set_ylim(-Ysim * 0.05, Ysim)
        ax.add_patch(Rectangle((R0, -Ysim * 0.05), (R1 - R0), Ysim * 0.05, edgecolor='k', facecolor='r'))
        ax.add_patch(Rectangle((R1, -Ysim * 0.05), (R2 - R1), Ysim * 0.05, edgecolor='k', facecolor='k'))
        ax.add_patch(Rectangle((R2, -Ysim * 0.05), (R3 - R2), Ysim * 0.05, edgecolor='k', facecolor='g'))
        ax.add_patch(Rectangle((R3, -Ysim * 0.05), (Rsim - R3), Ysim * 0.05, edgecolor='k', facecolor='k'))

        R_flat = np.linspace(R0, Rsim, num=500)
        R_i = R_flat[1] - R_flat[0]
        TYR_flux = np.zeros((len(R_flat), 3))
        TYR_flux[:, 2] = R_flat
        TYR_flux = tf.convert_to_tensor(TYR_flux, dtype=tf.float32)
        with tf.GradientTape() as g:
            g.watch(TYR_flux)
            C = network(TYR_flux)
        dC_dTYR = g.batch_jacobian(C, TYR_flux).numpy()
        Flux_density = dC_dTYR[..., 1].reshape(-1)
        Flux_density = Flux_density * FluxConvertToHale

        R_flat_disk = np.linspace(R0, R1, num=500)
        R_i_disk = R_flat_disk[1] - R_flat_disk[0]
        TYR_flux_disk = np.zeros((len(R_flat_disk), 3))
        TYR_flux_disk[:, 2] = R_flat_disk
        TYR_flux_disk = tf.convert_to_tensor(TYR_flux_disk)
        with tf.GradientTape() as g:
            g.watch(TYR_flux_disk)
            C = network(TYR_flux_disk)
        dC_dTYR_disk = g.batch_jacobian(C, TYR_flux_disk).numpy()
        Flux_density_disk = dC_dTYR_disk[..., 1].reshape(-1)
        J_D = sum(Flux_density_disk * R_flat_disk * R_i_disk)

        R_flat_ring = np.linspace(R2, R3, num=500)
        R_i_ring = R_flat_ring[1] - R_flat_ring[0]
        TYR_flux_ring = np.zeros((len(R_flat_ring), 3))
        TYR_flux_ring[:, 2] = R_flat_ring
        TYR_flux_ring = tf.convert_to_tensor(TYR_flux_ring)
        with tf.GradientTape() as g:
            g.watch(TYR_flux_ring)
            C = network(TYR_flux_ring)
        dC_dTYR_ring = g.batch_jacobian(C, TYR_flux_ring).numpy()
        Flux_density_ring = dC_dTYR_ring[..., 1].reshape(-1)
        J_R = sum(Flux_density_ring * R_flat_ring * R_i_ring)

        #Convert to the flux using W coordinates
        J_D = J_D + R0 ** 2 * 0.5 / FluxConvertToHale
        J_R = J_R
        J_D_Levich = R1 ** 2 * 0.5 / FluxConvertToHale

        collection_efficiency = - J_R / J_D

        collection_efficiency_AB = calcN_AB(R1, R2, R3)
        collection_efficiency_IL = calcN_IL(R1, R2, R3)

        ax.set_xlabel('R', fontsize='large', fontweight='bold')
        ax.set_ylabel('Y', fontsize='large', fontweight='bold')
        ax.set_title(
            r'$Sc^{\frac{1}{3}}Re^{\frac{1}{2}}$' + f'={Sc ** (1 / 3) * Re ** (1 / 2):.5f} RadialDiffusion={RADIALDIFFUSION}' + f'\n# n_corr_R={n_correction_R} n_corr_Y={n_correction_Y}' + f"freq={freq:.1f}Hz" + f"r1={r1:.2E}m r2={r2:.2E} m r3={r3:.2E}m")

        ax = axes[1]
        ax.plot(R_flat, Flux_density)

        ax.set_title(
            f"$J_D$={J_D:.4f},$J_R$={J_R:.4f},$J_{{D,Levich}}$={J_D_Levich:.4f},$N$={collection_efficiency:.2%},$N_{{AB}}$={collection_efficiency_AB:.2%},$N_{{IL}}$={collection_efficiency_IL:.2%}")

        ax.set_ylabel('Flux Density', fontsize='large', fontweight='bold')
        ax.set_xlabel('R', fontsize='large', fontweight='bold')

        fig.savefig(f'{saving_directory}/{file_name}.png', bbox_inches='tight')

        freqs.append(freq)
        r1s.append(r1)
        r2s.append(r2)
        r3s.append(r3)

        X_s.append(Sc ** (1 / 3) * Re ** (1 / 2))
        J_Ds.append(J_D)
        J_Rs.append(J_R)
        J_D_Levichs.append(J_D_Levich)
        Ns.append(collection_efficiency)
        NABs.append(collection_efficiency_AB)
        NILs.append(collection_efficiency_IL)
        n_corrections_list_R.append(n_correction_R)
        n_corrections_list_Y.append(n_correction_Y)

    tf.keras.backend.clear_session()
    plt.close('all')

    return f'./weights/weights {file_name}.h5'


def train_and_save(n_correction_R, n_correction_Y):
    saving_directory = f'Round 10h ID = {TASK_ID} epochs = {epochs} lambda_ratio = {lambda_ratio:.2f} radius_multiple = {radius_multiple:.2f} No Flux = {NO_FLUX_BND} RadialD = {RADIALDIFFUSION}'
    global initial_weights
    initial_weights = prediction(epochs=epochs, freq=freq, R0=0.05, r1=r1, r2=r2, r3=r3,
                                 lambda_ratio=lambda_ratio, alpha=0.98, n_correction_R=n_correction_R,
                                 n_correction_Y=n_correction_Y, num_train_samples=num_train_samples,
                                 batch_size=batch_size, train=False, initial_weights=initial_weights,
                                 saving_directory=saving_directory)
    sleep(randint(1, 20))

    df = pd.DataFrame({"freq": freqs, 'r1': r1s, "r2": r2s, "r3": r3s, 'X': X_s, "J_D": J_Ds, "J_R": J_Rs,
                       "J_D_Levichs": J_D_Levichs, 'n_corr_R': n_corrections_list_R,
                       "n_corr_Y": n_corrections_list_Y, "collection efficiency": Ns,
                       "collection efficiencyAB": NABs, "collection efficiencyIL": NILs})
    df.to_csv(f'{saving_directory}/results.csv', index=False)
    # df.to_csv(f'{results_directory}/Results ID = {SLRUM_ARRAY_TASK_ID} epochs = {epochs} freq = {freq:.2E} lambda_ratio = {lambda_ratio:.2f} radius_multiple = {radius_multiple:.2f} n_train={num_train_samples:.2E} n_samples = {n_samples:.2E} No Flux = {NO_FLUX_BND} RadialD = {RADIALDIFFUSION}.csv',index=False)


if __name__ == '__main__':

    maxT = 0.0  # maxT is zero as we want to solve a steady state problem independent of time
    epochs = 300

    #initial_weights = default_weight_name
    initial_weights = "weights/weights R0=5.00E-02 r1=1.00E-04 r2=1.10E-04 r3=1.44E-04 n_correction_R=0 n_correction_Y=0 lambda_ratio = 3.00E+00 maxY=3.00E+00 maxR=4.01E+00 alpha=0.98 epochs=3.00E+02 No Flux = False RadialD = True.h5"
    batch_size_sweep = [1000]  # a batchsize of 250 maybe too small.
    freq_sweep = [5.0]
    lambda_ratio_sweep = [3.0]  # Lambda_ratio should be larger than 1.5 to ensure a sufficiently distant boundary. 
    #radius_multiple_sweep = [1.0,2.5,4.5,5.0,7.5,10.0,15.0,20.0]

    radius_multiple_sweep = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
                             26.0, 27.0, 28.0, 29.0, 30.0]
    #radius_multiple_sweep = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
    num_train_samples_sweep = [int(8e6)]
    total_number_jobs = len(batch_size_sweep) * len(freq_sweep) * len(lambda_ratio_sweep) * len(
        radius_multiple_sweep) * len(num_train_samples_sweep)

    parameter_space = product(batch_size_sweep, freq_sweep, lambda_ratio_sweep, radius_multiple_sweep,
                              num_train_samples_sweep)
    parameter_space = list(parameter_space)

    for TASK_ID in range(0, 20):
        batch_size, freq, lambda_ratio, radius_multiple, num_train_samples = parameter_space[TASK_ID]
        r1 = 1e-5 * radius_multiple
        r2 = 1.1e-5 * radius_multiple
        r3 = 1.44e-5 * radius_multiple


        train_schedule = [
            [0, 0],
            [0, 1],
            [0, 0],
            [1, 0],
            [0, 0],
            [1, 1],
            [1, 2],
            [1, 1],
            [2, 1],
            [1, 1],
            [2, 2]
        ]

        for corrections in train_schedule:
            train_and_save(corrections[0], corrections[1])
