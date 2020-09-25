import numpy as np
#from quartercar import qc, roadprofile
#from tests import make_profile as mp
#from experiments import computeisoroughness as cisr
from matplotlib import pyplot as plt
import pandas as pd
from scipy.signal import periodogram, welch
from scipy.optimize import curve_fit
import logging





def qc_frequency_response(omega, epsilon, omega_s, omega_u, xi):
    """
    Returns the (acceleration) frequency response function of the quarter car model evaluated at frequencies. 
    :param omega: The angular frequencies at which to compute the PSD estimate
    :param epsilon: The ratio of sprung mass to unsprung mass (ms/mu)
    :param omega_s: The square root of the ratio of the suspension stiffness to the sprung mass (sqrt(ks/ms))
    :param omega_u: The square root of the ratio of the tire stiffness to the unsprung mass (sqrt(ku/mu))
    :param xi: The ratio of the suspension damping to the sprung mass and omega_s (cs/(2*ms*omega_s))
    :return: The value of the frequency response function H(f) evaluated at the frequencies with the input parameters
    """
    h_f = (2*omega_u**2* omega**2 * np.sqrt(omega_s**2)*np.sqrt(xi**2)*np.sqrt(omega**2 + omega_s**2/(4*xi**2)))/ \
        np.sqrt((2*omega*omega_u**2*omega_s*xi - 2*(epsilon+1)*omega**3*omega_s*xi)**2  + 
        (-1*omega**2*(epsilon*omega_s**2 + omega_u**2 + omega_s**2) + omega**4 + omega_u**2*omega_s**2 )**2)
    return h_f
    

def qc_objective_function(frequencies, epsilon, omega_s, omega_u, xi, C, w):
    """
    Defines the objective function for the (acceleration) PSD under the assumption that the 
    power spectral density of the road profile input is pink noise of the form: C*f-2
    :param frequencies: The frequencies at which to compute the PSD estimate
    :param epsilon: The ratio of sprung mass to unsprung mass (ms/mu)
    :param omega_s: The square root of the ratio of the suspension stiffness to the sprung mass (sqrt(ks/ms))
    :param omega_u: The square root of the ratio of the tire stiffness to the unsprung mass (sqrt(ku/mu))
    :param xi: The ratio of the suspension damping to the sprung mass and omega_s (cs/(2*ms*omega_s))
    :param C: The constant term representing the power/roughness of the road + the vehicle velocity
    :param w: The exponent that represents the PSD roughness coefficient of the road profile
    :return: The value of the function |H(f)|**2*C*f**-2 evaluated at the frequencies with the input parameters
    """
    omega = frequencies * 2*np.pi #frequencies need to be converted to angular frequency
    h_f = qc_frequency_response(omega, epsilon, omega_s, omega_u, xi)
    est_psd = h_f**2 * C * frequencies**-w
    return est_psd

def qc_jacobian(frequencies, epsilon, omega_s, omega_u, xi, C, w):
    """
    Defines the jacobian of the PSD under the assumption of a quarter car model with road profile
    input of pink noise
    :param frequencies: The frequencies at which to compute the PSD estimate
    :param epsilon: The ratio of sprung mass to unsprung mass (ms/mu)
    :param omega_s: The square root of the ratio of the suspension stiffness to the sprung mass (sqrt(ks/ms))
    :param omega_u: The square root of the ratio of the tire stiffness to the unsprung mass (sqrt(ku/mu))
    :param xi: The ratio of the suspension damping to the sprung mass and omega_s (cs/(2*ms*omega_s))
    :param C: The constant term representing the power/roughness of the road + the vehicle velocity
    :param w: The exponent term representing the roughness of the road profile
    :return: The jacobian of the function |H(f)|**2*C*f**-w w.r.t the input parameters
    """
    #shape of jacobian should be: len(frequencies), 5 
    #partial derivative of residuals w.r.t. xi
    #according to curve_fit, this should return: "the Jacobian matrix of the model function with respect to parameters as a dense array_like structure"
    #so this should return d_qc_objective_func/d_param, not d_residual/d_param (can test this)
    #term will be a part of each partial derivative (except for C)
    #TODO: Convert below to sympy?
    omega = frequencies * 2 * np.pi #convert to angular frequency for freq response function
    h_f = qc_frequency_response(omega, epsilon, omega_s, omega_u, xi)
    common_term = 2 * h_f *frequencies**-w*C

    #this term shows up in the numerator a lot
    num_term = omega**2 * omega_u**2 * np.sqrt(omega_s**2) * np.sqrt(xi**2) * np.sqrt(omega**2 + omega_s**2/(4*xi**2))

    #these terms show up in the denominator a lot
    dterm1 = (2*omega*omega_u**2*omega_s*xi - 2*(epsilon+1)*omega**3*omega_s*xi)

    dterm2 = (-omega**2*(epsilon*omega_s**2+omega_u**2 + omega_s**2) + omega**4 + omega_u**2*omega_s**2)
    

    #partial derivative w.r.t. epsilon
    df_deps = common_term * -((num_term * (-4*omega**3*omega_s*xi*dterm1 - 
    2*omega**2*omega_s**2*dterm2))/(dterm1**2 + dterm2**2)**(3/2))

    #partial derivative w.r.t. omega_s (assumes omega_s != 0)

    df_domega_s = common_term * ((2*(num_term * omega_s/(np.sqrt(omega_s**2))))/(np.sqrt(omega_s**2) * np.sqrt(dterm1**2 + dterm2**2)) + 
        (omega**2 * omega_u**2 * omega_s * np.sqrt(omega_s**2))/(2 * np.sqrt(xi**2) * np.sqrt(omega**2 + omega_s**2/(4*xi**2)) * np.sqrt(dterm1**2 + dterm2**2)) - 
            (num_term * (2 * dterm1/omega_s * dterm1  + 2*(2 * omega_u**2 * omega_s - omega**2*(2*epsilon*omega_s + 2*omega_s))*dterm2))/((dterm1**2 + dterm2**2)**(3/2)))
    
    #partial derivative w.r.t. omega_u (assumes omega_u != 0)
    df_domega_u = common_term * ((4*num_term/(omega_u))/(np.sqrt(dterm1**2 + dterm2**2)) - 
        (num_term * (8*omega*omega_u*omega_s*xi*dterm1 + 2*(2*omega_u*omega_s**2 - 2*omega**2*omega_u)*dterm2))/(dterm1**2 + dterm2**2)**(3/2))

    #partial derivative w.r.t. xi (assumes xi != 0)
    df_dxi = common_term * ((2*num_term*xi/np.sqrt(xi**2))/(np.sqrt(xi**2) * np.sqrt(dterm1**2 + dterm2**2))  - 
    (2*num_term* dterm1/xi * dterm1)/((dterm1**2 + dterm2**2)**(3/2)) - 
    (omega**2*omega_u**2*(omega_s**2)**(3/2)*np.sqrt(xi**2))/(2*xi**3*np.sqrt(omega**2 + omega_s**2/(4*xi**2)) * np.sqrt(dterm1**2 + dterm2**2)))

    #easy one, partial derivative w.r.t. C
    df_dc = h_f **2 * frequencies**-w

    #also easy, partial deriviative w.r.t. w, just multiply by the negative log of frequencies
    df_dw = C * h_f**2 * frequencies**-w * -np.log(frequencies)

    jacob = np.concatenate([df_deps.reshape(len(df_deps), -1), df_domega_s.reshape(len(df_domega_s), -1), df_domega_u.reshape(len(df_domega_u), -1),
    df_dxi.reshape(len(df_dxi), -1), df_dc.reshape(len(df_dc), -1), df_dw.reshape(len(df_dw), -1)], axis=1)

    return jacob

def fit(frequencies, psd, obj_func=qc_objective_function, jac=qc_jacobian, fmin=.1, fmax=30, params0=[4, 1, 10, .5, 2000*1e-8, 2], 
        sigma=None, bounds=None):
    """
    Fit finds the optimal parameters (in terms of least square fit) of the power spectral density under 
    the assumption that the PSD curve is the result of a quarter car model frequency response function 
    and road profile with pink noise PSD
    :param frequencies: The frequencies at which the PSD was computed
    :param psd: The PSD estimate at the given frequencies:
    :param obj_func: The objective function to use for fitting the curve to the data (default: qc_objective_func)
    :param jac: The function to use for computing the jacobian of the residuals with respect to the parameters (defualt: qc_jacobian)
    :param fmin: The minimum frequency to use when fitting the PSD curve (default: .1)
    :param fmax: The maximum frequency to use when fitting the PSD curve (defualt: 30)
    :param params: The parameters to use to pass into the curve_fitting method (defualt, assuming qc model: 
    epsilon0=ms/mu, 4; omega_s0 = sqrt(ks/ms), 1; omega_u0=sqrt(ku/mu), 10; xi0=cs/(2*ms*omega_s), .5;
    C0 = initial estimate of the velocity * C = 20 m/s * 100e-8)
    :param sigma: The estimate of the uncertainty in the psd estimate (default: None)
    :param bounds: The bounds of the parameters (default: None)
    :return: The optimal estimate of each of the paramters epsilon, omega_s, omega_u, xi, and C (although we don't really care about C)
    """
    frequencies, psd = frequencies[np.where(np.logical_and(frequencies >= fmin, frequencies<=fmax))], psd[np.where(np.logical_and(frequencies >= fmin, frequencies<=fmax))]
    #plt.loglog(frequencies, psd)
    #plt.show()
    if sigma is not None:
        sigma = sigma[np.where(np.logical_and(frequencies >= fmin, frequencies<=fmax))]
    if bounds is not None:
        popt, pcov = curve_fit(f=obj_func, xdata=frequencies, ydata=psd, p0=params0, sigma=sigma, bounds=bounds, jac=jac)
    else:
        #logging.debug("Sigma shape is {0}, psd shape is {1}".format(sigma.shape, psd.shape))
        popt, pcov = curve_fit(f=obj_func, xdata=frequencies, ydata=psd, p0=params0, sigma=sigma, jac=jac)
    return popt, pcov


#


#Steps:

#1. Will need to generate a road profile of a given length (longer the better)
#2. Will need to generate vehicle(s)
#3. Will need to drive over the profile at velocity(ies)
#4. Will then need to calculate the PSD using... welch's method?
#5. Will then need to fit the function using scipy's curve_fit method
#6. For this - need objective function (easy)
#7. Also, need method for computing jacobian (not as easy)
#8. Compare estimated parameters to true parameters