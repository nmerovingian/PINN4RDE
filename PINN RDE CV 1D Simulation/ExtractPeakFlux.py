import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd
import seaborn as sns 
import matplotlib.cm as cm


FD_directory = 'Paper Data Change Sigmas'
PINN_data_directory = 'Data'


sigmas = [1,2,5,10,15,20,30,40,50,60,70,80,90,100]
peak_fluxes = []

for index, sigma in enumerate(sigmas):
    PINN_file = f'{PINN_data_directory}/CV sigma={sigma:.2E} epochs=1.50E+02 n_train=1.00E+06.csv'
    df_PINN = pd.read_csv(PINN_file)

    peak_flux = df_PINN.iloc[:,1].min()
    peak_fluxes.append(peak_flux)


df = pd.DataFrame({'sigma':sigmas,'Peak Flux':peak_fluxes})
df.to_csv('PINN Peak Fluxes.csv',index=False)
