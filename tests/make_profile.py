from __future__ import division
import numpy as np



def make_sinusodal(wavelen, amplitude, prof_len, delta):
    """

    :param wavelen: The wavelength of the sinusoidal function (in units meters)
    :param amplitude: The amplitude of the sinusoidal function (in units mm)
    :param prof_len: The total length of the profile (in units meters)
    :param delta: The space between samples (in units meters)
    :return: A numpy array of profile elevation heights, and a numpy array of distance between points
    """
    num_samples = int(prof_len/delta)
    distances = np.linspace(0, prof_len, num_samples)
    prof_hts = amplitude*np.sin((2*np.pi/wavelen)*distances)
    return prof_hts, distances



def test_profile_func(x):
    if( x < 1):
        return 0
    elif(x >= 1 and x <=3):
        return x - 1
    elif(x >=3 and x <= 5):
        return 5 - x
    else:
        return 0

def test_profile(prof_len=11, delta=.25):
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
    disps = np.array(list(map(lambda x: test_profile_func(x), xs)))
    #print(disps)
    return disps, xs
    # print(disps)
   #slopes = np.diff(disps) / dx
    #print(slopes)