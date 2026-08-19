import numpy as np
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt
import math
PI = 3.14159265359
import scipy.integrate as integrate 


# Author : Farrukh - written for Ellie and now Jeff (Eid-ul-Fitr 2026 and Naurooz 2026)
# who knew I would repeat my thesis after 32 years

# parameter values these are used at the end of the code 
RISE_CONST = 0.02 # decay constant of t<0 decay
FALL_CONST = 0.5 # decay constant for t>0 decay
FRAC_RISE = 0.5 # fraction of t<0 decay
FRAC_FALL = 0.5 # fraction of t>0 decay  
SIGMA = 1e-6 # resolution for t>0 decay
# an exponential (lambda) convoluted with a Gaussian .. t>0 side
def fall_convolution(x, sigma, decayc):
    returnv1 = math.exp(pow(sigma,2)/(2*pow(decayc,2)) - x/decayc)
    returnv2 = math.erfc(-x/(math.sqrt(2)*sigma)+sigma/(math.sqrt(2)*decayc))
    return 1/(2*decayc)*returnv1*returnv2

# negative (t<0) side 
def rise_convolution(x, sigma, decayc):
    returnv1 = math.exp(pow(sigma,2)/(2*pow(decayc,2)) + x/decayc)
    returnv2 =1.0 + math.erf(-x/(math.sqrt(2)*sigma)-sigma/(math.sqrt(2)*decayc))
    return 1/(2*decayc)*returnv1*returnv2

def kernel(x:float, noise:float, rise_const:float, fall_const:float, frac_rise:float, frac_fall:float) -> float:
    return frac_rise * rise_convolution(x, noise, rise_const) + frac_fall * fall_convolution(x, noise, fall_const)

def mean_calculator(integration_limit:float, noise:float, rise_const:float, fall_const:float, frac_rise:float, frac_fall:float) -> float:
    def kernel_density(x):
        return kernel(x, noise, rise_const, fall_const, frac_rise, frac_fall)
    
    num, _ = integrate.quad(lambda x: x * kernel_density(x),
                                     -integration_limit,
                                     integration_limit,
                                     points=[0.0],
                                     limit=200)

    denom, _ = integrate.quad(kernel_density,
                                   -integration_limit,
                                   integration_limit,
                                   points=[0.0],
                                   limit=200)
    if denom <= 0.0:
        raise ValueError("Kernel area must be positive.")

    return float(num / denom)

def plot_pos_and_neg(x_values, sigma1, decayc1, frac1, sigma2, decayc2, frac2):
    l = np.size(x_values)
    values =np.zeros(l)
    # the functions for negative side and positive side are added up over the *entire*
    # negative and positive range 
    for i in range(0,l):
            values[i]=frac1*fall_convolution(x_values[i], sigma1, decayc1)
            values[i]+=frac2*rise_convolution(x_values[i], sigma2, decayc2)
                        
    return values

fig, ax1 = plt.subplots()
x_values = np.linspace(-2.0,2.0,1000)
mean = mean_calculator(10, SIGMA, RISE_CONST, FALL_CONST, FRAC_RISE, FRAC_FALL)
print("mean = {}".format(mean))

#ft = np.vectorize(plot_pos_and_neg)

print()

ax1.plot(x_values, plot_pos_and_neg(x_values, SIGMA, FALL_CONST, FRAC_FALL, SIGMA, RISE_CONST, FRAC_RISE))
plt.show()
         
