import numpy as np
import scipy
from matplotlib import pyplot as plt


def F_function(theta):
    return np.sqrt(3)/(4.0*np.pi)*np.log((1+theta**(1.0/3.0))**3/(1.0+theta)) + 3.0/(2.0*np.pi)*np.arctan((2.0*theta**(1/3)-1.0)/(np.sqrt(3))) + 1.0/4.0



def calcN_AB(r1,r2,r3):
    """
    r1: radius of disk electrode
    r2: internal radius of ring elctrode
    r3: outer radius of ring electrode
    """
    alpha = (r2/r1)**3 - 1.0
    beta = (r3/r1)**3 - (r2/r1)**3

    N = 1.0 - F_function(alpha/beta) + beta**(2.0/3.0)*(1.0-F_function(alpha)) - (1.0+alpha+beta)**(2.0/3.0)*(1.0-F_function((alpha/beta)*(1.0+alpha+beta)))

    return N



def calcN_IL(r1,r2,r3):
    alpha = (r2/r1)**3 - 1.0
    beta = (r3/r1)**3 - (r2/r1)**3

    N = 0.9674*(1.0-F_function((alpha+0.25)/beta))
    return N



if __name__ == "__main__":
    r1 = 0.348 # cm
    r2 = 0.386 # cm
    r3 = 0.4375 # cm

    alpha = (r2/r1)**3 - 1.0
    beta = (r3/r1)**3 - (r2/r1)**3
    print(alpha,beta)

    print(calcN_AB(r1,r2,r3),calcN_IL(r1,r2,r3))