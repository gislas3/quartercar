"""
This program will calculate the PSD numerically of road profiles, and match them to a roughness standard
according to ISO 8608 which can be found here: https://us.v-cdn.net/6030008/uploads/editor/83/oyhfu0i29vek.pdf
"""
import numpy as np
from scipy.signal import windows, fft
from quartercar import roadprofile as rp




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
    elevations = road_profile.gete_elevations()
    N = len(elevations)
    window = window_function(N)
    transform = fft(window*elevations)
    frequencies = np.linspace(0.0, 1.0/(2.0*dx), N//2)
    psd = np.abs(transform[0:N//2])**2
    return frequencies, psd

def smooth_psd(frequencies, psd):
    """
    Smooths the given PSD using the formula as defined by ISO 8608
    :param frequencies: The frequency bands, in cycles per meter, of the PSD
    :param psd: The original PSD
    :return: smoothed_psd: The smoothed PSD
    """
    pass







