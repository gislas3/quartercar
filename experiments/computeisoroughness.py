"""
This program will calculate the PSD numerically of road profiles, and match them to a roughness standard
according to ISO 8608 which can be found here: https://us.v-cdn.net/6030008/uploads/editor/83/oyhfu0i29vek.pdf
"""
import numpy as np
from scipy.signal import windows, welch
from scipy.fftpack import fft
from sklearn.linear_model import LinearRegression
from quartercar import roadprofile as rp


def create_profile_from_psd(road_type, method):
    """
    Creates a Road Profile from
    :param road_type:
    :param method:
    :return:
    """

def compute_psd(road_profile, dx, window_function=windows.hann):
    """
    Computes the Power Spectral Density of a road profile by method of FFT convolved with the window function
    specified
    :param road_profile: The `RoadProfile` instance whose PSD should be computed
    :param dx: The spacing between sample points in units m (assumes that profile has been evenly spaced prior
    to the function being called)
    :param: window_function: An function from the `scipy.signals.windows` library; the window function to use
    prior to computing the FFT
    :return: frequencies: The spatial frequencies in units cycles/meter of the PSD
    psd: The power spectrum over each frequency band
    """
    #Steps:
    #1. Compute FFT and
    elevations = road_profile.get_elevations()/1000
    sample_rate = 1/dx
    N = len(elevations)
    window = window_function(N)
    correction_factor = 1.63**2
    #frequencies, psd = welch(elevations, fs=sample_rate,  scaling='density')
    transform = fft(window*elevations*correction_factor)
    frequencies = np.linspace(0.0, 1.0/(2.0*dx), N//2)
    #print("Frequencies is {0}".format(frequencies))

    psd = (1/(sample_rate*N))*np.abs(transform[0:N//2])**2
    #print("psd is {0}".format(psd))
    return frequencies, psd

def smooth_psd(frequencies, psd):
    """
    Smooths the given PSD using the formula as defined by ISO 8608
    :param frequencies: The frequency bands, in cycles per meter, of the PSD
    :param psd: The original PSD
    :return: new_freqs: The frequency bands of the smoothed_psd.
            smoothed_psd: The smoothed PSD at each corresponding frequency
    """
    exp = -9
    step = 1
    new_freqs, smoothed_psd = [], []
    nl, nh = .0014, .0028
    be = frequencies[1] - frequencies[0] #original frequency resolution
    exp_step, nH_step = 1, 2
    #print("be is {0}".format(be))
    while exp <= 3:
        nL = int(nl/be + .5)
        nH = int(nh/be + .5)
        if nH > len(psd):
            break
        window_psds = np.sum(psd[nL+1:nH])*be
        smth_psd = (((nL + 0.5) * be - nl) * psd[nL] + window_psds + (nh - (nH - 0.5)*be)*psd[nH])/(nh - nl)
        new_freqs.append(2**exp)
        smoothed_psd.append(smth_psd)
        if exp == -5:
            step = 2/3
            nH_step = 2**(1/3)
        elif np.abs(exp - -4.333) <= 1e-3:
            step = 1/3
        elif np.abs(exp - -2) <= 1e-3:
            step = 1/12
            nh = .2726
            nH_step = 2**(1/12)
        nl = nh
        nh = nh*nH_step
        exp += step
    return np.array(new_freqs), np.array(smoothed_psd)

def fit_smooth_psd(frequencies, smooth_psd):
    """
    Performs a regression on the frequencies and smooth_psd to classify the road type, more info can be found
    on page 24-25
    :param frequencies: The frequency bands of the psd
    :param smooth_psd: The values of the smoothed psd at the frequency bands
    :return: The linear equation of best fit (least squares) of the frequencies to the smooth_psd
    """
    to_fit_inds = np.where(np.logical_and(frequencies <= 2.83, frequencies >= .011) > 0)
    lr = LinearRegression(fit_intercept=False)
    #print("Frequencies is {0}, smooth_psd is {1}".format(frequencies, smooth_psd))
    f_to_fit = frequencies[to_fit_inds]
    #print("smth psd is {0}".format(smooth_psd))
    spsd_to_fit = smooth_psd[to_fit_inds].reshape(-1, 1)
    #print("spsd to fit is {0}".format(spsd_to_fit))
    #print("F to fit is {0}".format(f_to_fit))
    f_to_fit = (f_to_fit)**(-2)

    #f_to_fit = f_to_fit.reshape(-1, 1)
    #print("F to fit is now {0}".format(f_to_fit))
    lr.fit(f_to_fit.reshape(-1, 1), spsd_to_fit)
    return lr



# Possible ways to simulate PSD/road roughness:

# 1. Use ISO standard, where PSD := Gd(n) = Gd(n0)*(n/n0)**-2; then integrate white noise?? issue: linear,
# PSD, not sure what it means by integrate white noise
# 2. Use Sun's method, where road profile := sum(A_k*cos(w_k*t + Phi_k) - problem is that this is temporal,
# not spatial, although could go back easily - 1 and 2 are the same thing!!!
# 3. Use either hybrid or homogeneous model from Models for road surface roughness paper,
# problem is I don't understand kernel or the PSD function they use





