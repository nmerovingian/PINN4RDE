import numpy as np
import pandas as pd

class Grid(object):
    def __init__(self,n):
        self.n = n
        self.Y_grid = np.linspace(start=0,stop=1.0,num=self.n,endpoint=True)
        self.conc = np.zeros(self.n)
        self.conc_d = np.zeros(self.n)

    def init_c(self,A:float):
        self.conc[:] = A
        self.conc_d = self.conc.copy()
    
    def grad(self):
        self.g = - (self.conc[1]-self.conc[0])/(self.Y_grid[1] - self.Y_grid[0])

        return self.g

    def update_d(self,Theta,concA):
        self.conc_d = self.conc.copy()
        self.conc_d[0] = 1.0/(1.0+np.exp(-Theta))
        self.conc_d[-1] = concA

    def save_conc_profile(self,file_name):
        df  = pd.DataFrame({'Y':self.Y_grid,'Conc':self.conc})
        df.to_csv(file_name,index=False)

    