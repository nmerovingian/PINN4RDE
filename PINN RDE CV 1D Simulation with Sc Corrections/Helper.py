import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def calcW(Y): 



    def dWdY(W,Y):
        dWdY = np.sqrt(1.65894)*np.exp(1.0/3.0*W**3)
        return dWdY

    Y_ = np.linspace(0,0.9999,num=100000,endpoint=False)
    W0 = 0.0

    W = odeint(dWdY,W0,Y_).reshape(-1)

    func_1D = interp1d(Y_,W)

    return func_1D(Y)


def calcY(W):
    def dWdY(W,Y):
        dWdY = np.sqrt(1.65894)*np.exp(1.0/3.0*W**3)
        return dWdY

    Y_ = np.linspace(0,0.9999,num=100000,endpoint=False)
    W0 = 0.0

    W_ = odeint(dWdY,W0,Y_).reshape(-1)

    func_1D = interp1d(W_,Y_)

    return func_1D(W)



if __name__ == '__main__': 
    """
    Y = np.random.rand(1000)*0.999
    W = calcW(Y)
    plt.scatter(Y,W)
    plt.xlabel('Y')
    plt.ylabel('W')
    plt.show()
    """


