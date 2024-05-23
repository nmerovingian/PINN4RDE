import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from lib.pinn import PINN
from lib.network import Network
from helper import ConcMontella 


class RDE2DDataset(Dataset):
    def __init__(self, num_train_samples, R0, re, no_flux_bnd):
        super().__init__()
        self.num_train_samples = num_train_samples
        self.R0 = R0
        self.re = re
        self.no_flux_bnd = no_flux_bnd

        # set up the parameters
        freq = 50  # Frequency of disc electrode
        omega = freq * np.pi * 2  # Rotational frequency
        nu = 1e-6  # Kinematic viscosity, s^-1
        D = 1e-9  # Diffusion coefficient, m^2 s^-1
        Sc = nu/D # Schmidt number
        Re = re**2*omega/nu  # Reynolds number
        L = 0.51023 * omega**1.5 * nu**(-0.5)
        delta = 1.2865 * (D / L)**(1/3)
        Delta = delta / re
        maxY = Delta * 4
        maxR = max(4, 1 + Delta * 4)
        script_L = L * re**3 / D

        self.L = L
        self.D = D
        self.Sc = Sc
        self.Re = Re
        self.script_L = script_L
        self.maxY = maxY
        self.maxR = maxR
        
        self.prepare_training_data()

    def prepare_training_data(self):
        # Solve the domain problem
        TYR_dmn0 = np.random.rand(self.num_train_samples, 3)
        TYR_dmn0[:, 0] = 0.0
        TYR_dmn0[:, 1] = TYR_dmn0[:, 1] * self.maxY
        TYR_dmn0[:, 2] = TYR_dmn0[:, 2] * (self.maxR - self.R0) + self.R0

        aux1_dmn0 = np.random.rand(self.num_train_samples, 1)
        aux1_dmn0[:, 0] = 1.0 / TYR_dmn0[:, 2]

        aux2_dmn0 = np.random.rand(self.num_train_samples, 1)  # scriptL * Y **2 
        aux2_dmn0[:, 0] = self.script_L * TYR_dmn0[:, 1] ** 2

        aux3_dmn0 = np.random.rand(self.num_train_samples, 1)  # scriptL * R * Y
        aux3_dmn0[:, 0] = self.script_L * TYR_dmn0[:, 2] * TYR_dmn0[:, 1]

        TYR_bnd0 = np.random.rand(self.num_train_samples, 3)
        TYR_bnd0[:, 0] = 0.0
        TYR_bnd0[:, 1] = 0.0
        TYR_bnd0[:, 2] = TYR_bnd0[:, 2] * (1.0 - self.R0) + self.R0

        TYR_bnd1 = np.random.rand(self.num_train_samples, 3)
        TYR_bnd1[:, 0] = 0.0
        TYR_bnd1[:, 1] = 0.0
        TYR_bnd1[:, 2] = TYR_bnd1[:, 2] * (self.maxR - 1.0) + 1.0

        TYR_bnd2 = np.random.rand(self.num_train_samples, 3)
        TYR_bnd2[:, 0] = 0.0
        TYR_bnd2[:, 1] = TYR_bnd2[:, 1] * self.maxY
        TYR_bnd2[:, 2] = self.maxR

        TYR_bnd3 = np.random.rand(self.num_train_samples, 3)
        TYR_bnd3[:, 0] = 0.0
        TYR_bnd3[:, 1] = self.maxY
        TYR_bnd3[:, 2] = (self.maxR - self.R0) * TYR_bnd3[:, 2] + self.R0

        TYR_bnd4 = np.random.rand(self.num_train_samples, 3)
        TYR_bnd4[:, 0] = 0.0
        TYR_bnd4[:, 1] = TYR_bnd4[:, 1] * self.maxY
        TYR_bnd4[:, 2] = self.R0

        C_dmn0 = np.zeros((self.num_train_samples, 1))
        C_bnd0 = np.zeros((self.num_train_samples, 1))
        C_bnd1 = np.zeros((self.num_train_samples, 1))
        C_bnd2 = np.ones((self.num_train_samples, 1))
        C_bnd3 = np.ones((self.num_train_samples, 1))
        if self.no_flux_bnd:
            C_bnd4 = np.zeros((self.num_train_samples, 1))
        else:
            C_bnd4 = ConcMontella((self.L / self.D) ** (1 / 3) * TYR_bnd4[:, 1] * self.re)
            C_bnd4 = C_bnd4.reshape(-1, 1)

        self.inputs = [TYR_dmn0, aux1_dmn0, aux2_dmn0, aux3_dmn0, TYR_bnd0, TYR_bnd1, TYR_bnd2, TYR_bnd3, TYR_bnd4]
        self.outputs = [C_dmn0, C_bnd0, C_bnd1, C_bnd2, C_bnd3, C_bnd4]
    
    def __len__(self):
        return len(self.inputs[0])
    
    def __getitem__(self, idx):
        input_data = [torch.tensor(data[idx], dtype=torch.float32) for data in self.inputs]
        output_data = [torch.tensor(data[idx], dtype=torch.float32) for data in self.outputs]
        return input_data, output_data
    

def test_fn(network, R0, maxR, maxY, saving_directory, file_name, Sc, Re, FluxConvertToHale,
            num_test_samples, device):
    """
    Test the network to plot concentration profiles and calculate flux densities at a specified time step.

    Args:
        network: Trained model to predict concentration using PyTorch.
        R0, maxR, maxY: Parameters defining the range for R and Y.
        saving_directory: Directory where to save the outputs.
        file_name: Base name for saved files.
        Sc, re: Schmidt and Reynolds numbers for the current setup.
        FluxConvertToHale: Conversion factor for flux density.
    """
    network.eval()
    network = network.to(device)
    time_sects = [0.0]
    
    
    # results
    X_s = []
    J_sses = []
    R0_s = []

    for index, time_sect in enumerate(time_sects):
        Y_flat = np.linspace(0, maxY, num_test_samples)
        R_flat = np.linspace(R0, maxR, num_test_samples)
        R_i = R_flat[1] - R_flat[0]
        Y, R = np.meshgrid(Y_flat, R_flat)
        XYR_test = torch.zeros((int(num_test_samples**2), 3), dtype=torch.float32, device=device)
        XYR_test[..., 0] = torch.tensor(time_sect)
        XYR_test[..., 1] = torch.tensor(Y.flatten())
        XYR_test[..., 2] = torch.tensor(R.flatten())
        X = XYR_test[..., 0]

        # Predict concentration
        with torch.no_grad():
            C = network(XYR_test).reshape(Y.shape)
        
        C = C.cpu().numpy()
        np.save(f'{saving_directory}/{file_name}.npy', np.array([R, Y, C]))

        # Plotting the concentration profile
        fig, axes = plt.subplots(figsize=(8, 9), nrows=2)
        plt.subplots_adjust(hspace=0.4)
        ax = axes[0]
        mesh = ax.pcolormesh(R, Y, C, shading='auto')
        cbar = plt.colorbar(mesh, pad=0.05, aspect=10, ax=ax)
        cbar.set_label('C(Y,R)')
        cbar.mappable.set_clim(0, 1)
        ax.set_ylim(-maxY * 0.05, maxY)
        ax.add_patch(Rectangle((R0,-maxY*0.05),(1.0-R0),maxY*0.05,edgecolor='k',facecolor='r'))
        ax.add_patch(Rectangle((1.0,-maxY*0.05),(maxR-1.0),maxY*0.05,edgecolor='k',facecolor='k'))
        ax.set_xlabel('R', fontsize='large', fontweight='bold')
        ax.set_ylabel('Y', fontsize='large', fontweight='bold')
        ax.set_title(r'$Sc^{\frac{1}{3}}Re^{\frac{1}{2}}$' + f'={Sc**(1/3)*Re**(1/2):.5f}')

        # Calculate flux density
        R_flat = np.linspace(R0, maxR,num=500)
        R_i = R_flat[1] - R_flat[0]
        TYR_flux = torch.zeros((len(R_flat), 3), dtype=torch.float32, device=device)
        TYR_flux[..., 2] = torch.tensor(R_flat, dtype=torch.float32, device=device)
        TYR_flux.requires_grad_(True)
        with torch.enable_grad():
            C = network(TYR_flux)
            gradients = torch.autograd.grad(outputs=C, inputs=TYR_flux, grad_outputs=torch.ones_like(C), create_graph=True)[0]
        Flux_density = gradients[..., 1].detach().cpu().numpy()
        
        J_ss =  sum(Flux_density*R_i)
        # Convert to the flux using W coordinates
        J_ss = J_ss * FluxConvertToHale + R0

        C = C.detach().cpu().numpy().flatten()
        # Plotting flux density
        ax = axes[1]
        ax.plot(R_flat, Flux_density)
        ax.set_title(f'J={J_ss:.4f}')
        ax.set_ylabel('Flux Density', fontsize='large')
        ax.set_xlabel('R-axis', fontsize='large')

        # Save figures and data
        fig.savefig(f'{saving_directory}/{file_name}.png', bbox_inches='tight')
        np.save(f'{saving_directory}/{file_name}_flux_density.npy', np.array([R_flat, Flux_density]))

        plt.close(fig)  # Close the figure to free memory

        X_s.append(Sc**(1/3)*Re**(1/2))
        J_sses.append(J_ss)
        R0_s.append(R0)

    return X_s, J_sses, R0_s


def prediction(epochs=50, maxT=0, R0=0.5, re=1e-5, num_train_samples=int(2e6),
               weights_path=None, train=True, saving_directory="./Data", alpha=0.98,
               no_flux_bnd=False, device='cuda'):
    
    Path(saving_directory).mkdir(parents=True, exist_ok=True)
    Path("./weights").mkdir(parents=True, exist_ok=True)

    # set training hyperparameters
    num_train_samples = int(num_train_samples)
    num_test_samples = 1000
    batch_size = 250

     # init dataset 
    train_dataset = RDE2DDataset(num_train_samples, R0, re, no_flux_bnd=no_flux_bnd)

    # get parameters
    maxT = maxT
    maxY = train_dataset.maxY
    maxR = train_dataset.maxR
    file_name = f'R0={R0:.2E} re={re:.2E} maxY={maxY:.2E} maxR={maxR:.2E} alpha={alpha:.2f} no_flux = {no_flux_bnd} epochs={epochs:.2E} n_train={num_train_samples:.2E}'

    # define network
    network = Network()
    
    # define PINN
    pinn = PINN(network, no_flux_bnd)
    if weights_path is not None:
        pinn.load_state_dict(torch.load(weights_path))
    pinn = pinn.to(device)
    
    optimizer = optim.Adam(pinn.parameters(), lr=1e-3)  # In tensorflow, the default lr of Adam optimizer is 1e-3
    
    # set scheduler
    def lr_lambda(epoch):
        if epoch <= 50:
            return 1
        else:
            return alpha ** (epoch - 50)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    if weights_path is None:
        weights_path = f'./weights/weights {file_name}.h5'

    # Training loop
    if train:
        pinn.train()
        for i_epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {i_epoch+1}/{epochs}")
            losses = []
            for inputs, outputs in train_loader:
                inputs = [inp.to(device) for inp in inputs]
                outputs = [out.to(device) for out in outputs]

                optimizer.zero_grad()
                preds = pinn(inputs)
                mse_losses = [F.mse_loss(pred, target) for pred, target in zip(preds, outputs)]
                loss = torch.stack(mse_losses).sum()
                losses.append(loss.item())

                loss.backward()
                optimizer.step()
                lr = optimizer.param_groups[0]['lr']

                pbar.set_postfix({'Loss': loss.item(), 'lr': lr})
                pbar.update()

            mean_loss = np.mean(losses)
            scheduler.step()
            print(f'Epoch {i_epoch+1}/{epochs}, Loss: {mean_loss:.6f}, lr: {lr}')
            pbar.close()

        torch.save(pinn.state_dict(), weights_path)
        return weights_path
    else:  # eval loop
        # load weights
        pinn.load_state_dict(torch.load(weights_path)) 
        pinn.eval()

        L = train_dataset.L
        D = train_dataset.D
        Sc = train_dataset.Sc
        Re = train_dataset.Re

        FluxConvertToHale = np.sqrt(1.65894)/(((L/D)**(1/3))*re)

        # results
        X_s, J_sses, R0_s = test_fn(network, R0=R0, maxR=maxR, maxY=maxY, saving_directory=saving_directory,
                                    file_name=file_name, Sc=Sc, Re=Re, FluxConvertToHale=FluxConvertToHale, 
                                    num_test_samples=num_test_samples, device=device)
        return X_s, J_sses, R0_s


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    maxT = 0.0 # maxT is zero as we want to solve a steady state problem independent of time
    epochs = 150
    no_flux_bnd = False
    train = False

    saving_directory = f'Data No Flux = {no_flux_bnd}'
    weights_path = None
    X_s, J_sses, R0_s = [], [], []

    for R0 in [0.05]:
        for num_train_samples in [3e6]:
            for re in np.arange(7e-6, 4.3e-5, step=2e-6):
                if train:
                    prediction(epochs=epochs, R0=R0, re=re, alpha=0.98, num_train_samples=num_train_samples, train=True, no_flux_bnd=no_flux_bnd,
                                              weights_path=weights_path, saving_directory=saving_directory, device=device)
                else:
                    X, J_ss, R0 = prediction(epochs=epochs, R0=R0, re=re, alpha=0.98, num_train_samples=num_train_samples, train=False, no_flux_bnd=no_flux_bnd,
                                                   weights_path=weights_path,saving_directory=saving_directory, device=device)
                    X_s.extend(X)
                    J_sses.extend(J_ss)
                    R0_s.extend(R0)
    if not train:
        df = pd.DataFrame({'X':X_s,"J_ss":J_sses,'R0':R0_s})
        df.to_csv(f'{saving_directory}/results.csv',index=False)
