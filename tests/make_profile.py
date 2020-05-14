from __future__ import division
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import multivariate_normal
from scipy.signal import butter, sosfilt
from matplotlib import pyplot as plt


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
    orig_hts = multivariate_normal.rvs(cov=np.eye(num_samples)*sigma)
    sr = 1/(distances[1] - distances[0])
    #print("Shape of heights is {0}".format(orig_hts.shape))
    orig_hts = np.random.normal(0, sigma, num_samples)
    a = np.exp(-2*np.pi*cutoff_freq*delta)
    b = 1 - np.exp(-2*np.pi*cutoff_freq*delta)
    prof_hts = np.zeros(len(orig_hts))
    for x in range(0, len(prof_hts)): #low pass filter
        if(x > 0):
            prof_hts[x] = b*orig_hts[x] + a*prof_hts[x-1]

    #sos = butter(2, 3/(sr/2), 'lowpass', fs=sr, output='sos')
    #prof_hts = sosfilt(sos, orig_hts)
    cs = CubicSpline(distances, prof_hts)
    new_dists = np.linspace(0, profile_len, int(profile_len/delta2))
    new_heights = cs(new_dists)
    return distances, orig_hts, prof_hts, new_dists, new_heights


def iso_psd_function(g_n0, n):
    """
    Defines the iso psd function given g_n0 (assumes units of frequency cycles per meter)
    :param g_n0: The value of G_d(n0)
    :param n: The frequency at which to calculate the value of the function
    :return: The function evaluated at n
    """
    n0 = .1
    return g_n0*1e-6 * (n/n0)**(-2)

def make_profile_from_psd(road_type, method, delta, prof_len, seed=55):
    """
    Creates a road profile from a PSD function, using the method specifed by the caller
    :param road_type: The class of the road type, according to ISO 8608
    :param method: The methodology to use for
    :param delta: The spacing between successive profile points (in meters)
    :param prof_len: The length of the profile (in meters)
    :return: distances: The spatial points at which the road profile is defined
             heights: The elevations of the road profile (in mm)
    """
    np.random.seed(seed)
    #For now, we're going to use the ISO 8608 standard for the PSD function
    if road_type == 'A':
        lower, upper = 1, 32
        mean = 16
    elif road_type == 'B':
        lower, upper = 32, 128
        mean = 64
    elif road_type == 'C':
        lower, upper = 128, 512
        mean = 256
    elif road_type ==  'D':
        lower, upper = 512,  2048
        mean = 1024
    elif road_type == 'E':
        lower, upper = 2048, 8192
        mean = 4094
    elif road_type == 'F':
        lower, upper = 8192, 32768
        mean = 16384
    elif road_type == 'G':
        lower, upper = 32768,131072
        mean = 65536
    else: #road_type == 'H': - if it's called with garbage, just give a rough road
        lower, upper = 131072, 524288
        mean = 262144
    g_n0 = min(upper, max(lower, np.random.normal(loc=mean, scale=mean/4)))
    print("GNO is {0}".format(g_n0))
    if method == 'hybrid': #TODO: Implement code for hybrid methodology
        pass
    else: #Use the cosine/sine method - might be a bit confusing since we're going to need switch back and forth between angular frequency
        freq_min, freq_max = 1e-5, 15#this is going to be defined in cycles/meter, just going to convert at each step
        #According to https://deepblue.lib.umich.edu/bitstream/handle/2027.42/769/77008.0001.001.pdf?sequence=2 (page 39),
        #the wavenumbers should go from .0033 cycles/ft to 1 cycle/ft (so .01 cycles/meter to 3.28 cycles/meter)
        #also, the separation at "low" frequencies should be .0033 cycle/ft (so .01 cycle/meter) and at "high" .01 cycle/ft
        #(so .0328 cycles/meter); we're going to do a few extra frequencies; high frequencies we'll define as anything
        #over .5 cycles/meter (no justification for this honestly)
        freqs = []
        f = freq_min
        freq_step = .01
        while f < freq_max:
            freqs.append(f)
            f += freq_step
            if f >= .5:
                freq_step = .0328
        freqs = np.array(freqs)
       # M = len(freqs)
        freq_deltas = np.concatenate((np.array([freq_min]), (freqs[2:] - freqs[:-2])/2, np.array([(freqs[-2] - freqs[-4])/2])))
        #print("Freq deltas is {0}".format(freq_deltas))
        distances = np.arange(0, prof_len + delta, delta)
        elevations = np.zeros(len(distances))
        psd_vals = np.array(list(map(lambda x: iso_psd_function(g_n0, x), freqs)))
        phase_angles = np.random.uniform(-np.pi, np.pi, len(freqs))
        for x in range(0, len(elevations)):
            d = distances[x]
            elevations[x] = np.sum(np.sqrt(2*freq_deltas*psd_vals)*np.cos(2*np.pi*freqs*d + phase_angles))
        elevations = elevations - np.mean(elevations) #so it has zero mean

        return distances, elevations*1000






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
    xs = np.linspace(0, prof_len, num_samples)
    #print(xs)
    disps = np.array(list(map(lambda x: iri_test_profile_func(x), xs)))
    #print(disps)
    return xs, disps*10
    # print(disps)
   #slopes = np.diff(disps) / dx
    #print(slopes)