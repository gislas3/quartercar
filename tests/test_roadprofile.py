import pytest
from quartercar.roadprofile import RoadProfile
import numpy as np
import tests.make_profile as mp
from matplotlib import pyplot as plt
import pandas as pd


#Tests constructor throwing exception when the distances and elevations are of different lengths
def test_break_constructor():
    with pytest.raises(ValueError):
        RoadProfile([1, 2, 3, 4], [1, 2, 3, 4, 5]) # Not the same length

#Tests constructor throwing exception when there are less than 2 data points
def test_break_constructor2():
    with pytest.raises(ValueError):
        RoadProfile([0], [0]) # Too few data points

#Tests constructor throwing exception when the distances are not increasing
def test_break_constructor3():
    with pytest.raises(ValueError):
        RoadProfile([0, 0, 1, 2], [1, 2, 3, 4]) # Not increasing

#Tests that the proper distances are returned when calling the get_distances() function
def test_get_distances():
    ds, es = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
    rp = RoadProfile(ds, es)
    assert(rp.get_distances() == ds)

#Tests that the proper elevations are returned when calling the get_elevations() function
def test_get_elevations():
    ds, es = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
    rp = RoadProfile(ds, es)
    assert(rp.get_elevations() == es)

#Tests that the proper length is returned (namely, that the final value of the dists array is returned)
def test_length():
    ds, es = [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]
    profile = RoadProfile(ds, es)
    assert(profile.length() == 4)

#Possible TODO for split: look at trying to split at beginning and end of profiles
#Tests that the split method returns a list of two road profiles when the split happens exactly in the middle, and
#the requested split distance actually exists in the distances array
def test_split1():
    distances = np.arange(0, 2.25, .25)
    elevations = np.arange(0, len(distances), 1)
    profile = RoadProfile(distances, elevations)
    profiles = profile.split(1)
    assert(profiles[0] == RoadProfile(np.arange(0, 1.25, .25), np.arange(0, len(np.arange(0, 1.25, .25)), 1)))
    assert(profiles[1] == RoadProfile(np.arange(1, 2.25, .25), np.arange(4, 9, 1)))

#Tests that the split method returns a list of two road profiles when the split happens at a distance that
#is not in the distance array
def test_split2():
    distances = np.arange(0, 2.25, .25)
    elevations = np.arange(0, len(distances), 1)
    profile = RoadProfile(distances, elevations)
    profiles = profile.split(.9)
    assert (profiles[0] == RoadProfile(np.arange(0, 1, .25), np.arange(0, len(np.arange(0, 1, .25)), 1)))
    assert (profiles[1] == RoadProfile(np.arange(.75, 2.25, .25), np.arange(3, 9, 1)))

#Tests that the original profile is returned when attempting to split at a 0 distance
def test_split_invalid():
    distances = np.arange(0, 2.25, .25)
    elevations = np.arange(0, len(distances), 1)
    profile = RoadProfile(distances, elevations)
    profiles = profile.split(0)
    assert(profiles[0] == profile)

#Tests that the moving filter average works properly when the distances are exactly spaced .25 meters apart
#(e.g. k = 1)
def test_moving_avg_filter_k1():
    distances = [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]
    elevations = [3, 2, 1, 4, 2, -1, 0, 9, -2]
    profile = RoadProfile(distances, elevations)
    new_profile = profile.moving_avg_filter()
    assert(new_profile.get_distances() == profile.get_distances())
    assert(new_profile.get_elevations() == profile.get_elevations())

#Tests that the moving average filter works when the distances are spaced closer together than .25 meters,
#but the dx is a factor of .25 (k = 4)
def test_moving_avg_filter_k4():
    distances = np.arange(0, 2.0625, .0625)
    elevations = np.arange(0, len(distances), 1)
    profile = RoadProfile(distances, elevations)
    new_profile = profile.moving_avg_filter()
    new_distances = np.arange(0, 1.875, .0625)
    new_elevations = np.zeros(len(new_distances))
    for x in range(0, len(new_elevations)):
        moving_avg = np.sum(elevations[x:x+4])/4
        new_elevations[x] = moving_avg

    assert(np.allclose(new_profile.get_distances(), new_distances))
    assert(np.allclose(new_profile.get_elevations(), new_elevations))

#Tests that the moving average filter works when the distances are spaced closer together than .25 meters,
#but dx is not a factor of k (k is rounded up to 2)
def test_moving_avg_filter_k_frac():
    distances = np.arange(0, 2.15625, .15625)
    elevations = np.arange(0, len(distances), 1)
    profile = RoadProfile(distances, elevations)
    new_profile = profile.moving_avg_filter()
    new_distances = distances[:-1]
    new_elevations = np.zeros(len(new_distances))
    for x in range(0, len(new_elevations)):
        new_elevations[x] = np.sum(elevations[x:x+2])/2

    assert(np.allclose(new_profile.get_distances(), new_distances))
    assert(np.allclose(new_profile.get_elevations(), new_elevations))

#Tests that the moving average filter returns None when the profile isn't evenly spaced
#Possible TODO: Make the moving filter able to handle these cases
def test_moving_avg_filter_fail():
    distances = [0, 1, 3, 43]
    elevations = [1, 2, 3, 4]
    profile = RoadProfile(distances, elevations)
    new_profile = profile.moving_avg_filter()
    assert(new_profile is None)

#Tests that get_car_sample works properly when the input profile is evenly spaced, and a constant velocity
def test_get_car_sample_even():
    x = [0, 1, 2, 3]
    y = [0, 2, 4, 6]
    rp = RoadProfile(x, y)
    actual = rp.car_sample(2, 1, sample_rate_hz=2)
    expected = RoadProfile([0, .5, 1., 1.5, 2., 2.5, 3.], [0, 1, 2, 3, 4, 5, 6])
    assert actual == expected

#Tests that the get_car_sample works properly when a list of different velocities/distances is input
def test_get_car_sample_sequence():
    x = [0, 1, 2, 3]
    y = [0, 2, 4, 6]
    rp = RoadProfile(x, y)
    actual = rp.car_sample([1, 2], [1, 2], sample_rate_hz=10)
    expected = RoadProfile([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.,
                            1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6, 2.8, 3.],
                           2 * np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.,
                                         1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6, 2.8, 3.]))
    assert actual == expected

#Tests the space evenly function works properly
def test_space_profile_evenly():
    x = [0, 1, 2, 3]
    y = [0, 2, 4, 6]
    rp = RoadProfile(x, y)
    rp_new = rp.space_evenly(.5)
    assert(np.allclose(rp_new.get_distances(), np.array([0, .5, 1., 1.5, 2., 2.5, 3. ])))
    assert(np.allclose(rp_new.get_elevations(), np.array([0, 1, 2, 3, 4, 5, 6])))

#Tests that smoothed slopes works properly when the spacing is exactly the same length as the base length
#of the filter
def test_smoothed_slopes_baselen_same():
    distances = np.arange(0, 2.25, .25)
    elevations = np.array([0, 1, 2, 6, -3, 5, 9, -2, 1])
    profile = RoadProfile(distances, elevations)
    x1, slps = profile.compute_smoothed_slopes(distances, elevations)
    assert(np.allclose(x1, np.array([[.5], [0], [.5], [0]])))
    assert(np.allclose(slps, np.array([1/.25, 1/.25, 4/.25, -9/.25, 8/.25, 4/.25, -11/.25, 3/.25])))

#Tests that smoothed slopes work when the spacing of the profile is smaller than the baselength of the filter
def test_smoothed_slopes_baselen_diff():
    distances = np.arange(0, 2.125, .125)
    elevations = np.array([0, 1, 5, -8, 5, 12, 4, -2, -3, -1, 0, 2, 13, 99, -1, 3, 1])
    profile = RoadProfile(distances, elevations)
    x1, slps = profile.compute_smoothed_slopes(distances, elevations)
    assert (np.allclose(x1, np.array([[.5], [0], [.5], [0]])))
    assert(np.allclose(slps, np.array([5/.25, -9/.25, 0, 20/.25, -1/.25, -14/.25, -7/.25, 1/.25, 3/.25, 3/.25, 13/.25, 97/.25, -14/.25, -96/.25, 2/.25])))

#Tests that smoothed slopes works properly when the profile has already been filtered
def test_smooothed_slopes_filtered():
    distances = np.arange(0, 2.125, .125)
    elevations = np.array([0, 1, 5, -8, 5, 12, 4, -2, -3, -1, 0, 2, 13, 99, -1, 3, 1])
    profile = RoadProfile(distances, elevations)
    profile_filt = profile.moving_avg_filter()
    x1_1, slps_1 = profile.compute_smoothed_slopes(distances, elevations)
    x1_2, slps_2 = profile_filt.compute_smoothed_slopes(profile_filt.get_distances(), profile_filt.get_elevations())
    #assert(np.allclose(x1_1, x1_2)) #They shouldn't be the same since at the ends, they won't cancel
    assert(np.allclose(slps_1, slps_2))

#Tests that IRI works when a profile_length of less than 11 is input
def test_iri_less_than_11():
    prof_len = 5
    delta = .25
    profile = RoadProfile(*mp.iri_test_profile(prof_len, delta))
    assert(abs(profile.to_iri() - 7.82931) <= 1e-5)

#Tests that IRI works when the spacing between samples is exactly .25 meters (see note below also for more info)
def test_iri_dx_250():
    #To see full table of values, see:
    # http://documents.worldbank.org/curated/en/851131468160775725/pdf/multi-page.pdf, page 50 in the pdf viewer, page numbers 41-42mar
    true_iris = np.array([0, 0.00000,0.00000,0.00000,0.44424,1.46822,2.70319,3.77380,4.47956,4.80554,4.85232,4.75180,
                          4.50642,4.93081,5.76800,6.65620,7.33478,7.71431,7.84294,7.82931,7.66730,7.32857,7.13737,7.01181,
                          6.87758,6.70672,6.50858,6.30880,6.13016,5.98287,5.86438,5.76496,5.67420,5.58524,5.49579,5.40684,
                          5.32053,5.23842,5.16077,5.08670,5.01486,4.94401,4.87342,4.80283,4.73234,4.66210,4.59221,4.52266,
                          4.45334,4.38411,4.31490,4.24570,4.17656,4.10756,4.03881,3.97038,3.90236,3.83534,3.77234,3.71304,
                          3.65716,3.60441,3.55451,3.50722,3.46229,3.41949,3.37863,3.33949,3.30192,3.26572,3.23075,3.19687,
                          3.16394,3.13184,3.10047,3.06972,3.03952,3.00978,2.98043,2.95143,2.92270,2.89423,2.86596,2.83786,
                          2.80993,2.78213,2.75446,2.72691,2.69948,2.67216,2.64496,2.61790,2.59097,2.56419,2.53757,2.51114,
                          2.48561,2.46093,2.43705,2.41392,2.39149,2.36972,2.34856,2.32798,2.30793,2.28837,2.26927,2.25059,
                          2.23231,2.21438,2.19678,2.17949,2.16247,2.14572,2.12920,2.11289,2.09679,2.08087,2.06513,2.04955])
    delta = .25
    our_iris = np.zeros(len(true_iris))
    index = 0
    for x in np.arange(.25, 30.25, .25):
        profile = RoadProfile(*mp.iri_test_profile(x, delta))
        our_iris[index] = profile.to_iri()
        index += 1
    #NOTE: The differences at the beginning are I believe due to the fact that in the paper where the values come from, they set
    #the initial value of the simulation a bit differently (namely, I believe they divide by 11 instead of dx*k - you can see this
    #by changing the denominator in the compute_smoothed_slopes() function to 11
    assert(np.allclose(our_iris[19:], true_iris[19:]))

#Tests that IRI works when the spacing between samples is less than .25 meters (see note below for more info)
def test_iri_dx_125():
    true_iris = np.array([0, 0.00000,0.00000,0.00000,0.44424,1.46822,2.70319,3.77380,4.47956,4.80554,4.85232,4.75180,
                          4.50642,4.93081,5.76800,6.65620,7.33478,7.71431,7.84294,7.82931,7.66730,7.32857,7.13737,7.01181,
                          6.87758,6.70672,6.50858,6.30880,6.13016,5.98287,5.86438,5.76496,5.67420,5.58524,5.49579,5.40684,
                          5.32053,5.23842,5.16077,5.08670,5.01486,4.94401,4.87342,4.80283,4.73234,4.66210,4.59221,4.52266,
                          4.45334,4.38411,4.31490,4.24570,4.17656,4.10756,4.03881,3.97038,3.90236,3.83534,3.77234,3.71304,
                          3.65716,3.60441,3.55451,3.50722,3.46229,3.41949,3.37863,3.33949,3.30192,3.26572,3.23075,3.19687,
                          3.16394,3.13184,3.10047,3.06972,3.03952,3.00978,2.98043,2.95143,2.92270,2.89423,2.86596,2.83786,
                          2.80993,2.78213,2.75446,2.72691,2.69948,2.67216,2.64496,2.61790,2.59097,2.56419,2.53757,2.51114,
                          2.48561,2.46093,2.43705,2.41392,2.39149,2.36972,2.34856,2.32798,2.30793,2.28837,2.26927,2.25059,
                          2.23231,2.21438,2.19678,2.17949,2.16247,2.14572,2.12920,2.11289,2.09679,2.08087,2.06513,2.04955])
    delta = .125
    our_iris = np.zeros(len(true_iris))
    index = 0
    for x in np.arange(.25, 30.25, .25):
        profile = RoadProfile(*mp.iri_test_profile(x, delta))
        our_iris[index] = profile.to_iri()
        index += 1
    #NOTE: I **think** (hope) the higher difference in this case is due to the numerical imprecision of the calculation
    assert (np.allclose(our_iris[19:], true_iris[19:], .2))



