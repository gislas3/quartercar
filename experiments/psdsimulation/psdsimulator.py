import logging 

import numpy as np

from quartercar import cars, roadprofile
from tests import make_profile

class PSDSimulator():

    def __init__(self, n_profiles, profile_length, sample_rate_hz, seed=None):
        self.n_profiles = n_profiles
        self.profile_length = profile_length
        self.sample_rate_hz = sample_rate_hz
        self.rng = np.random.default_rng(seed)

    def get_random_car(self):
        # if car == '':
        #     self.car = cars.get_
        # try:
        #     self.car = cars.get_car_dict()[car]
        # except Exception as e:
        #     logging.warning("Tried to set car to invalid value of {0}, randomly selecting a different valid car".format(
        #         car
        #     ))
        return cars.get_random_car(self.rng)
    
    def get_random_device_orientation(self):
        yaw = self.rng.uniform(-180, 180) # hard coding these limits for now
        pitch = self.rng.uniform(-80, 80)
        roll = self.rng.uniform(-45, 45)
        return yaw, pitch, roll

    def get_random_device_misalignment(self): # This does not include the gains
        angles = self.rng.normal(0, 2.5*np.pi/180, 6) # hard coding for now
        return np.array([[0, angles[0], angles[1]], 
        [angles[2], 0, angles[3]], [angles[4], angles[5], 0]])

    def run_simulation(self):
        # Steps: 
        # get the car 
        # simulate the device orientation w.r.t. car
        # simulate the device misalignment angles
        # 
        car = self.get_random_car()
        device_yaw, device_pitch, device_roll = self.get_random_device_orientation()
        misalignment_mat = self.get_random_device_orientation()
        for x in range(0, self.n_profiles):
            pass
            # Steps:
            # Get road profile with a given length and PSD (randomly simulate A, B, C, D, E, etc. road)
            # get the accelerometer bias
            # get the accelerometer gain
            # get the accelerometer noise sigma
            # get the velocity
            # Compute the series of x, y, and z accelerations with the given info
            # then... 
            # run function for computing: 
            # pitch and roll angles
            # then... 
            # run function for computing: 
            # "true" z accelerations
            # then...
            # run function for estimating transfer function
            # then...
            # estimate PSD of road profile by dividing by the modulus squared of PSD estimate of transfer function
            # 
        # once everything is done, compute mean transfer function? TBD

if __name__ == 'main':
    pass