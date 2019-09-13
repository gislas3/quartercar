from quartercar.roadprofile import RoadProfile
import pytest
import numpy as np


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



    