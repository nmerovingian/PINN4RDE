import math
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def calcW(U):
    # This function converts U (which is after Hale transformation) to W.
    def dWdU(W,U):
        dWdU = np.sqrt(1.65894)*np.exp(1.0/3.0*W**3)
        return dWdU
    U_ = np.linspace(0,0.99999,num=100000,endpoint=False)
    W0 = 0.0
    W = odeint(dWdU,W0,U_).reshape(-1)
    interp_1D = interp1d(U_,W,fill_value='extrapolate') 
    return interp_1D(U)



if __name__ == '__main__': 
    U = np.linspace(0,0.9999,num=1000)
    W = calcW(U)

    plt.plot(U,W)
    plt.xlabel('U',fontsize='large',fontweight='bold')
    plt.ylabel('W',fontsize='large',fontweight='bold')
    plt.savefig('U-W.png',bbox_inches='tight')


