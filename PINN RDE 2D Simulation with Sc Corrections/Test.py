import numpy as np     

re = np.array([1e-5,1e-4,1e-3,1e-2,1e-1])

freq = 50 #freq of disc electrode 
omega = freq * np.pi*2 # Rotational freq
nu = 1e-6 #s^-1 kinematic viscosity
L =  0.51023*omega**1.5*nu**(-0.5)


D = 1e-9 # m^2 s^-1, diffusion coefficients 

script_L = L * re**3 / D # The dimensionless form of L

print(script_L)