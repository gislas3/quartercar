
import numpy as np
from scipy.fftpack import fft
from scipy.signal import periodogram
from scipy.signal import welch
from scipy.linalg import expm, inv

class RoadProfile():

    def __init__(self, distances, elevations):
        """
        :param distances: A list of distance from the starting point of the profile (in units meters)
        :param elevations: A list of elevations (in mm)
        :raises: ValueError if the distances are not increasing and any two distances have the same value
        :raises: ValueError if the elevations and distances are *not* the same length
        :raises: ValueError if there are less than 2 data points 
        """
        self.elevations = elevations
        self.distances = distances

        if not len(elevations) == len(distances):
            raise ValueError(f'Expected length of elevations (${len(elevations)}) and distances (${distances}) to be equal')

        if len(elevations) < 2:
            raise ValueError(f'Expected at least two data points but received ${len(elevations)}')

        if not np.all(np.diff(distances) > 0):
            #print("Np.diff(distances) > 0 is {0}".format(np.diff(distances) > 0 ))
            #print("All Np.diff(distances) > 0 is {0}".format(np.all(np.diff(distances) > 0)))
            #print("Diffs are 0 at indices {0}".format(np.where(np.diff(distances) <= 0)))
            #print(distances[np.where(np.diff(distances) == 0)])
            #print("distances 50+ are {0}".format(distances[50:]))
            #print("Diffs are {0}".format(np.diff(distances)))
            raise ValueError('All distances must be increasing and not the same')
        

    def __eq__(self, other):
        if not isinstance(other, RoadProfile):
            return False

        return len(self.distances) == len(other.distances) and \
                np.allclose(self.distances, other.distances) and \
                np.allclose(self.elevations, other.elevations)


    def __repr__(self):
        return f'RoadProfile(${self.distances}, ${self.elevations})'


    def get_elevations(self):
        return self.elevations


    def get_distances(self):
        return self.distances


    def car_sample(self, distances, velocities, sample_rate_hz=100):
        """
        This function returns a sample of its road profile from a car driving on it at a constant speed or range of speed
        :param velocities: Either an array or an float. If an array is given it is assumed to contain the instantaneous velocity of the car over specified distances (in m/s). If a float is given, it is assumed to be a fixed velocity for all road profile points.
        :param distances: Either an array or an float. If an array is given it is assumed to contain a one-to-one mapping of distance traveled at each specified velocity in velocities (in m). If a float is given the data points are assumed to be evenly spaced with the given distance.
        :return: A RoadProfile that represents the points of the profile that was sampled by a car traveling at specified velocities (in m/s), with specified sample rate (in Hz)
        """

        new_profile = None

        # Create evenly spaced 
        if isinstance(velocities, (int, float)) and isinstance(distances, (int, float)):
            delta_x = velocities/sample_rate_hz
            new_profile = self.space_evenly(delta_x)

        elif hasattr(velocities, '__len__') and hasattr(distances, '__len__'):
            if not len(velocities) == len(distances):
                raise ValueError(f'Distances and velocities are not the same length (${len(distances)} vs ${len(velocities)})')

            #What we want to do:
            #1. Get distance for each sample rate of the vehicle
            #2. Get elevation for each distance point
            #3. Get times based on each instantaneous velocity
            #4. Interpolate evenly spaced profile based on either min distance or average distance bet samples
            #5. Compute final times, distances, and elevations, return
            #velocities, dists = vel_dists[0], vel_dists[1]
            dist_list, time_list = [], []
            total_dist = 0#, total_time = 0, 0
            for x in range(0, len(velocities)):
                vel, dist = velocities[x], distances[x]
                time = dist/vel #time in seconds it took to travel that interval
                num_samples = time*sample_rate_hz #total number of samples over that interval
                for y in range(0, int(num_samples)):
                    dist_list.append(total_dist)
                    total_dist += dist/num_samples
            dist_list.append(total_dist)
            new_dists = np.array(dist_list)
            new_elevations = np.interp(new_dists, self.distances, self.elevations)
            new_profile = RoadProfile(new_dists, new_elevations)
        else:
            raise ValueError('Velocities and distances are expected to have the same type, but got: ' + type(velocities) + ' ' + type(distances))

        #if velocities is not None: #can just compute time for each distance
        #    pass
            #get sample points where the car would actually be traveling on the road

        #elif vel_dists is not None:
            
        return new_profile


    def space_evenly(self, distance=None):
        """
        Returns a new `RoadProfile` where the distances are evenly spaced.
        :param distance: The length in mm desired between road elevation measurements
        :return: A new immutable RoadProfile object with evenly spaced distances
        """

        if distance is None:
            min_dist = min(np.min(np.diff(self.distances)), .25) #for now, we are going to interpolate at min distance or 300 mm
            #TODO: Probably need some check in case there are a few distances closely spaced, and others not so
        else:
            min_dist = min(distance, self.length())
        number_of_samples = int(self.length() / min_dist) + 1
        #print("Number of samples is {0}, length is {1}, min_dist is {2}".format(number_of_samples, self.length(), min_dist))
        final_dists = np.linspace(0, self.length(), number_of_samples)
        final_elevs = np.interp(final_dists, self.distances, self.elevations)
        return RoadProfile(final_dists, final_elevs)


    def length(self):
        """
        :return: The length of the entire profile in meters
        """
        return self.distances[-1]

    def split(self, distance):
        """
        :param distance: The distance in meters in which the road profile should be split
        :return: A list of RoadProfile split via distance
        """
        pass


    def compute_smoothed_slopes(self, base_len=.25, init_length=11):
        """

        :param base_len: The baselength of the moving average filter to compute the smoothed road profile, in meters (default = .25 meters = 250 mm)
        :init_length: The length of the initial profile as to which to initialize the IRI equation
        :return: A numpy array of smoothed profile height slopes
        """
        list_slps = []
        total_dist = 0
        x1 = None
        prev_dist = 0
        total_dist = 0
        for x in range(0, len(self.elevations)):  # compute the moving average filter of the slopes

            denom = self.distances[x] - prev_dist
            total_dist += denom
            if(total_dist < init_length and total_dist + self.distances[x] >= init_length):
                x1 = np.array([[(self.elevations[x] - self.elevations[0])/total_dist], [0], [(self.elevations[x] - self.elevations[0])/total_dist ], [0]])
                #print(x1)
            next_ind = x + 1

            while(next_ind < len(self.elevations) and denom < base_len):
                denom += self.distances[next_ind] - self.distances[next_ind-1]
                next_ind += 1

            if(next_ind < len(self.elevations)):
                smoothed_slope = (self.elevations[next_ind] - self.elevations[x])/denom
                #print("Smoothed slope is {0}".format(denom))
                list_slps.append(smoothed_slope)
            prev_dist = self.distances[x]
        #print("List slps is {0}".format(list_slps))
        #TODO: Make check to ensure that sample rate is high enough
        return x1, np.array(list_slps)


    def to_iri(self):
        """
        :return: The IRI value over the entire road profile
        """
        # parameters taken from Sayers paper, correspond to Golden Car simulation parameters, more details
        # can be found here: http://onlinepubs.trb.org/Onlinepubs/trr/1995/1501/1501-001.pdf
        c = 6.0
        k_1 = 653
        k_2 = 63.3
        mu = 0.15

        a_mat = np.array(
            [[0, 1, 0, 0], [-k_2, -c, k_2, c], [0, 0, 0, 1], [k_2 / mu, c / mu, -(k_1 + k_2) / mu, -c / mu]])
        inv_a = inv(a_mat)
        b_mat = np.array([[0], [0], [0], [k_1 / mu]])
        vel = 80 / 3.6  # convert simulated speed of 80 km/hr to m/s
        curr_pro_len = 0  # keep track of the estimated distance traveled along the profile
        # TODO: optimize calculation to not use matrix math
        # C = np.array([[1, 0, -1, 0]])
        # iri_dict = {}  # store the computed iri values
        curr_sum = 0  # current numerator for the calculation
        n = 0  # current denominator for the calculation
        #assumption is that we've already been evenly spaced
        dx = np.diff(self.distances)[0]

        x1, smooth_slps = self.compute_smoothed_slopes()
        if(x1 is not None):
            ex_a = expm(a_mat * dx / vel)  # just use .25 as sample rate
            term2 = np.matmul(np.matmul(inv_a, (ex_a - np.eye(4))), b_mat)

            for ind in range(0, len(smooth_slps)):  # again, for more details, see Sayers paper
                slp = smooth_slps[ind]
                if (ind == 0):
                    x = x1
                else:  # see Sayers paper
                    term1 = np.matmul(ex_a, x)
                    x = term1 + term2 * slp
                #print(x)
                curr_sum += abs(x[0, 0] - x[2, 0])
                n += 1

        to_ret_iri = None
        if (n > 0 and curr_sum > 0):
            to_ret_iri = min(curr_sum / n, 20)
        return to_ret_iri


