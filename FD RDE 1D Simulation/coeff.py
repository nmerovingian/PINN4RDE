import numpy as np
from helper import calcW


class Coeff(object):
    def __init__(self,n,deltaT,Y_grid):
        self.n = n

        self.aA = np.zeros(self.n)
        self.bA = np.zeros(self.n)
        self.cA = np.zeros(self.n)
        self.deltaT = deltaT
        self.Y_grid = Y_grid
        self.deltaY = Y_grid[1] - Y_grid[0]
        self.A_matrix = np.zeros((self.n,self.n))




    def Acal_abc(self,deltaT,deltaY):
        self.A_matrix[0,0] = 0.0
        self.A_matrix[0,1] = 1.0


        for i in range(1,self.n-1):
            W = calcW(self.Y_grid[i])
            self.A_matrix[i,i-1] = -np.exp(-2.0/3.0*W**3)/1.65894 * self.deltaT / (self.deltaY * self.deltaY)
            self.A_matrix[i,i] = 2.0 * np.exp(-2.0/3.0*W**3)/1.65894 * self.deltaT / (self.deltaY * self.deltaY) + 1.0 
            self.A_matrix[i,i+1] = -np.exp(-2.0/3.0*W**3)/1.65894 * self.deltaT / (self.deltaY * self.deltaY)


        self.A_matrix[-1,-1] = 1.0



        