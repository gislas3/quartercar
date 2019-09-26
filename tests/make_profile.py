from __future__ import division
import numpy as np
from scipy.interpolate import CubicSpline


def make_sinusodal(wavelen, amplitude, profile_len, delta):
    """

    :param wavelen: The wavelength of the sinusoidal function (in units meters)
    :param amplitude: The amplitude of the sinusoidal function (in units mm)
    :param prof_len: The total length of the profile (in units meters)
    :param delta: The space between samples (in units meters)
    :return: A numpy array of profile elevation heights, and a numpy array of distance between points
    """
    num_samples = int(profile_len/delta)
    distances = np.linspace(0, profile_len, num_samples)
    #print("Distances is {0}".format(distances))
    prof_hts = amplitude*np.sin((2*np.pi/wavelen)*distances)
    return distances, prof_hts


def make_gaussian(sigma, profile_len, delta, cutoff_freq, delta2=None, seed=None):
    """
    For methodology, see page 186 of: http://systemdesign.illinois.edu/publications/All08b.pdf
    :param sigma: The standard deviation of profile height for the gaussian distribution (in mm)
    :param profile_len: The length of the profile (in meters)
    :param delta: The space between samples of the original profile (in meters)
    :param cutoff_freq: The cutoff spatial frequency to use in the filtering step (low pass filter)
    :param delta2: The ultimate spacing of the final profile (in meters - for use in the final smoothing step)
    :param seed: The seed to use for the random number generator (default: None, means use no seed)
    :return: Two arrays, one representing the distanace, another representing the profile heights
    """
    np.random.seed(seed)
    num_samples = int(profile_len/delta)
    distances = np.linspace(0, profile_len, num_samples)
    orig_hts = np.random.normal(0, sigma, num_samples)
    a = np.exp(-2*np.pi*cutoff_freq*delta)
    b = 1 - np.exp(-2*np.pi*cutoff_freq*delta)
    prof_hts = np.zeros(len(orig_hts))
    for x in range(0, len(prof_hts)): #low pass filter
        if(x > 0):
            prof_hts[x] = b*orig_hts[x] + a*prof_hts[x-1]
    cs = CubicSpline(distances, prof_hts)
    new_dists = np.linspace(0, profile_len, int(profile_len/delta2))
    new_heights = cs(new_dists)
    return distances, orig_hts, prof_hts, new_dists, new_heights



def iri_test_profile_func(x):
    if( x < 1):
        return 0
    elif(x >= 1 and x <=3):
        return x - 1
    elif(x >=3 and x <= 5):
        return 5 - x
    else:
        return 0



def iri_test_profile(prof_len=11, delta=.25):
    #The rationale + table for this can be found at:
    # http://documents.worldbank.org/curated/en/851131468160775725/pdf/multi-page.pdf, page 50 in the pdf viewer, page numbers 41-42mar
    # 25 meter long formula:
    # y = 0, x < 1
    # y = x - 1, x>= 1 x <= 3
    # y = 5 - x, x>=3 and x <=5
    # y=0, x >= 5
    # let's do: delta = .25
    # let's try meters first
    #dx = 0.125
    num_samples = int(prof_len/delta) + 1
    #print(num_samples)
    xs = np.linspace(0, 11, num_samples)
    #print(xs)
    disps = np.array(list(map(lambda x: iri_test_profile_func(x), xs)))
    #print(disps)
    return disps, xs
    # print(disps)
   #slopes = np.diff(disps) / dx
    #print(slopes)