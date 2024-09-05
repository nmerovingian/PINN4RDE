import numpy as np 
from coeff import Coeff
from grid import Grid
from scipy import sparse
from scipy.sparse import linalg
from matplotlib import pyplot as plt
import time
import pandas as pd 
from concurrent.futures import ProcessPoolExecutor
import os


bulk_A = 1.0



def simulation(sigma):

    #start and end potential of voltammetric scan
    theta_i =10.0
    theta_v = -10.0
    cycles = 1 # bynver if cyckes=
    deltaY = 1e-3
    n = int(1.0/deltaY+1.0) 
    deltaTheta = 1e-2  # potential step of simulation

    deltaT = deltaTheta/sigma
    maxT = 2.0*abs(theta_v-theta_i)/sigma

    grid = Grid(n)
    grid.init_c(bulk_A)
    coeff = Coeff(n,deltaT,grid.Y_grid)
    coeff.Acal_abc(deltaT=deltaT,deltaY=deltaY)


    #simulation steps
    nTimeSteps = int(2*np.fabs(theta_v-theta_i)/deltaTheta)
    Esteps = np.arange(nTimeSteps)
    E = np.where(Esteps<nTimeSteps/2.0,theta_i-deltaTheta*Esteps,theta_v+deltaTheta*(Esteps-nTimeSteps/2.0))
    E = np.tile(E,cycles)
    start_time = time.time()


    CV_location = f'{directory}/sigma={sigma:.6E}'

    fluxes = []
    for index in range(0,int(len(E))):
        Theta = E[index]
        
        if index == 10:
            print(f'Total run time is {(time.time()-start_time)*len(E)/60/10:.2f} mins')
        
        grid.conc = linalg.spsolve(sparse.csr_matrix(coeff.A_matrix),sparse.csr_matrix(grid.conc_d[:,np.newaxis]))
        grid.update_d(Theta=Theta,concA=bulk_A)
        flux = grid.grad()
        fluxes.append(flux)

        """
        # Save concentration profile at the end of forward scan. 
        if index == int(len(E)/2):
            grid.save_conc_profile(f'{CV_location} conc {index/len(E):.2f}.csv')
        """

    # Save the voltammogram
    df = pd.DataFrame({'Potential':E,'Flux':fluxes})
    df.to_csv(f'{CV_location}.csv',index=False)

if __name__ =='__main__':
    global directory 
    directory = f'FD Simulation Results'

    if not os.path.exists(directory):
        os.mkdir(directory)

    for sigma in [0.001,0.01,0.1,0.5,1,10,30,100,1000]:
        simulation(sigma)

        