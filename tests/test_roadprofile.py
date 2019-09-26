from quartercar.roadprofile import RoadProfile
import pytest
import numpy as np
import tests.make_profile as mp
from matplotlib import pyplot as plt
import pandas as pd


def test_break_constructor():
    with pytest.raises(ValueError):
        RoadProfile([1, 2, 3, 4], [1, 2, 3, 4, 5]) # Not the same length

def test_break_constructor():
    with pytest.raises(ValueError):
        RoadProfile([0], [0]) # Too few data points

def test_break_constructor(): 
    with pytest.raises(ValueError):
        RoadProfile([0, 0, 1, 2], [1, 2, 3, 4]) # Not increasing

def test_space_profile_evenly():
    x = [0, 1, 2, 3]
    y = [0, 2, 4, 6]
    rp = RoadProfile(x, y)
    rp_new = rp.space_evenly(.5)
    np.allclose(rp_new.get_distances(), np.array([0, .5, 1., 1.5, 2., 2.5, 3. ]))
    np.allclose(rp_new.get_elevations(), np.array([0, 1, 2, 3, 4, 5, 6]))

def test_get_car_sample_even():
    x = [0, 1, 2, 3]
    y = [0, 2, 4, 6]
    rp = RoadProfile(x, y)
    actual = rp.car_sample(2, 1, sample_rate_hz=2)
    expected = RoadProfile([0, .5, 1., 1.5, 2., 2.5, 3. ], [0, 1, 2, 3, 4, 5, 6])
    assert actual == expected
    
def test_get_car_sample_sequence():
    x = [0, 1, 2, 3]
    y = [0, 2, 4, 6]
    rp = RoadProfile(x, y)
    actual = rp.car_sample([1, 2], [1, 2], sample_rate_hz=10)
    expected = RoadProfile([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1., 
                            1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6, 2.8, 3.], 
                           2*np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1., 
                            1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6, 2.8, 3.]))
    assert actual == expected

def test_smoothed_slopes():
    pass

def test_gaussian_profile():
    sigma = 8 #8 mm
    profile_len, delta, cutoff_freq, delta2, seed = 100, .1, .15, .01, 55
    dists, orig_hts, low_pass_hts, final_dists, final_heights = mp.make_gaussian(sigma, profile_len, delta, cutoff_freq, delta2, seed)
    show_plots = False
    if show_plots:
        plt.plot(dists, orig_hts)
        plt.title("Original heights/distances")
        plt.show()

        plt.plot(dists, low_pass_hts)
        plt.title("Low pass filtered heights with cutoff spatial frequency of {0}".format(cutoff_freq))
        plt.show()

        plt.plot(final_dists, final_heights)
        plt.title("Plot of final dists/heights")
        plt.show()

        rp = RoadProfile(final_dists, final_heights)
    #print(rp.to_iri())
    #assert(False)
    #data_dict = {"Distances": final_dists, "Elevations": final_heights}
    #df = pd.DataFrame(data_dict)
    #df.to_csv("/Users/gregoryislas/Documents/Mobilized/test_profile.csv", index=False)
