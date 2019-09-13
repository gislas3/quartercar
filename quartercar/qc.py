import numpy
from .roadprofile import RoadProfile
import numpy as np
from scipy import signal

class QC():
    """
    A quarter car model that represents the interplay between a road surface (`RoadProfile`), a fixed (and always grounded) mass and a sprung mass.
    The class 
      1) calculates acceleration values from a `RoadProfile` and velocities (see `run`) and 
      2) reverses the calculation to generate a `RoadProfile` from accelerations, distances and velocities.
    """

    def __init__(self, c_s=0, k_s=0, k_l=0, m_s=0, m_u=0): #TODO: Put in all different car parameters (with defaults) as constructor arguments
        """
        :param c_s: The suspension damping rate (assumes units N*s/m)
        :param k_s: The suspension spring constant (assumes units N/m)
        :param k_l: The tire spring constant (assumes units N/m)
        :param m_s: The sprung mass (assumes units kg)
        :param m_u: The unsprung mass (assumes units kg)
        """
        #self.c_s = c_s #suspension damping rate (in N * s/m)
        #self.k_s = k_s #suspension spring constant (in units N/m)
        #self.k_l = k_l #tire spring constant (in units N/m)
        #self.m_s  = m_s #sprung mass (in units kg)
        #self.m_u = m_u #unspring mass (in units kg)
        #TODO: Check to make sure parameters valid
        self.m_s = m_s
        #divide coefficients by sprung mass to make for easier computation into IRI algorithm
        #Can always get back originals by multiplying by m_s
        self.c = c_s/self.m_s
        self.k1 = k_l/self.m_s
        self.k2 = k_s/self.m_s
        self.mu = m_u/self.m_s


    #need a method here to estimate initial state
    def get_initial_state(self, road_profile):
        """
        :param road_profile: The `RoadProfile` instance that contains the information about the road profile in question
        :return: An estimate of the initial state of the state space
        """
        #for now, we're just gonna return the way the IRI algorithm does it, average profile change over the first 11 meters
        elevations, dists = road_profile.get_elevations(), road_profile.get_distances()
        inds = np.where(dists >= 11)
        x0 = np.array([[0], [0], [0], [0]])
        if(len(inds[0]) > 0): #if the profile is longer than 11 meters, use x0
            #just return 0
            ind0 = inds[0][0]
            init_val = (elevations[ind0] - elevations[0])/dists[ind0]
            x0 = np.array([[init_val], [0], [init_val], [0]])
        return x0

    def get_time_array(self, road_profile, velocities):
        """

        :param road_profile: The `RoadProfile` instance that contains the information about the road profile in question
        :param velocities: The instantaneous velocity of the car at each point of the road profile
        :return:
        """
        elevations, dists = road_profile.get_elevations(), road_profile.get_distances()
        diffs = np.diff(dists)
        dts = diffs/velocities






    def run(self, road_profile, velocities, distances, sample_rate_hz=100):
        """
        This is where we generate the acceleration values from the road profile, running the entire QC simulation

        :param road_profile: An array or list like of floating point values (in mm) of road profile elevations
        :param velocities: Either an array or an float. If an array is given it is assumed to contain the instantaneous velocity of the car over specified distances (in m/s). If a float is given, it is assumed to be a fixed velocity for all road profile points.
        :param distances: Either an array or an float. If an array is given it is assumed to contain a one-to-one mapping of distance traveled at each specified velocity in velocities (in m). If a float is given the data points are assumed to be evenly spaced with the given distance.
        :return: list of acceleration values (in m/s^2)
        """
        #sample_rate_hz = 100
        a = np.array(
            [[0, 1, 0, 0], [-self.k2, -self.c, self.k2, self.c], [0, 0, 0, 1], [self.k2 / self.mu, self.c / self.mu, -(self.k1 + self.k2) / self.mu, -self.c / self.mu]])

        b = np.array([[0], [0], [0], [self.k1 / self.mu]])
        c = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-self.k2, -self.c, self.k2, self.c]])
        #More information about the state space can be found in http://onlinepubs.trb.org/Onlinepubs/trr/1995/1501/1501-001.pdf
        #Basically, we just want dx/dt = Ax + bu, y = x
        state_space = signal.StateSpaceContinuous(a, b, c, np.array([[0]]))
        # for now, we are going to use distance instead of time, and we are going to assume that it's already evenly spaced
        # otherwise, do something like below
        #rp_cleaned = road_profile.clean()
        #TODO: Appropriate error checking below

        # Get the points touched by the car at the given velocity, distance car traveled at velocity, and sample rate
        road_sample = road_profile.get_car_sample(velocities, distances, sample_rate_hz)

        #if using average dx:
        dx = np.average(np.diff(road_sample.get_distances()))
        #if using min dx:
        #dx = np.min(np.diff(road_sample.get_distances()))
        total_time = np.sum(distances/velocities)
        road_sample = road_sample.clean(dx)
        velocity = road_sample.length()/total_time
        # should be even since road profile sampled evenly
        
        times = np.cumsum(np.diff(road_sample.get_distances()) / velocity)
        x0 = self.get_initial_state(road_sample)

        final_states = signal.lsim(state_space, road_sample.get_elevations(), times, x0)

        return final_states





        #
        #return accs

    def inverse(self, accelerations, distances, velocities):

        """


        :param accelerations: An array of vertical accelerations in (m/s^2) (assumes that preprocessing step,
            subtracting gravity/accounting for orientation has already occurred)
        :param distances: An array that contains the distance (in mm) from start of measurements
        :param velocities: An array that contains the instantaneous velocity of the car at each distance

        :return: `RoadProfile`
        """

        #return road_profile
        pass






