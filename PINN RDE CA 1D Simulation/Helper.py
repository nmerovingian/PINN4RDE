import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def calcW(Y): 
    D = 1e-9 #m^2 s^-1, diffusion coefficient of water
    nu  = 1e-6 # m^2 s^-1, kinematic viscosity of water
    f = 50 # Hz rotational speed


    L = 0.51023*(2*math.pi*f)**(1.5)*nu**(-0.5)


    def dWdY(W,Y):
        dWdY = np.sqrt(1.65894)*np.exp(1.0/3.0*W**3)
        return dWdY

    Y_ = np.linspace(0,0.9999,num=100000,endpoint=False)
    W0 = 0.0

    W = odeint(dWdY,W0,Y_).reshape(-1)

    func_1D = interp1d(Y_,W)

    return func_1D(Y)



if __name__ == '__main__': 
    Y = np.random.rand(1000)*0.999
    W = calcW(Y)
    plt.scatter(Y,W)
    plt.xlabel('Y')
    plt.ylabel('W')
    plt.show()


