
import numpy as np
from scipy.fftpack import fft
from scipy.signal import periodogram
from scipy.signal import welch
from scipy.linalg import expm, inv

class RoadProfile():

    def __init__(self, elevations, distances):
        """

        :param elevations: A list of elevations (in mm)
        :param distances: A list of distance from the starting point of the profile (in units meters)
        """

        self.elevations = elevations
        self.distances = distances
        #print("Self.elevations is {0}, self.distances is {1}".format(self.elevations, self.distances))

    def get_elevations(self):
        return self.elevations

    def get_distances(self):
        return self.distances


    def get_car_sample(self, velocity=None, vel_dists=None, sample_rate_hz=100):
        """
        This function returns a sample of its road profile from a car driving on it at a constant speed or range of speed
        :param velocity: Should be set if the car traveled at a constant velocity over the entire road profile (should be m/s)
        :param velocities: Should be a list of velocities (in m/s) if there is an instantaneous velocity at each point in the road profile
        (e.g. the car had different velocities over the course of the profile)
        :return: A list of times, distances, and elevations that can be used in a QC simulation
        """

        new_profile = None
        if(velocity is not None): #can just compute time for each distance
            delta_x = velocity/sample_rate_hz
            new_profile = self.clean(delta_x) #get sample points where the car would actually be traveling on the road

        elif(vel_dists is not None):
            #What we want to do:
            #1. Get distance for each sample rate of the vehicle
            #2. Get elevation for each distance point
            #3. Get times  based on each instantaneous velocity
            #4. Interpolate evenly spaced profile based on either min distance or average distance bet samples
            #5. Compute final times, distances, and elevations, return
            velocities, dists = vel_dists[0], vel_dists[1]
            dist_list, time_list = [], []
            total_dist = 0#, total_time = 0, 0
            for x in range(0, len(velocities)):
                vel, dist = velocities[x], dists[x]
                time = dist/vel #time in seconds it took to travel that interval
                num_samples = time*sample_rate_hz #total number of samples over that interval
                for y in range(0, int(num_samples)):
                    dist_list.append(total_dist)
                    total_dist += 1
            new_dists = np.array(dist_list)
            new_elevations = np.interp(np.array(dist_list), self.distances, self.elevations)
            new_profile = RoadProfile(new_dists, new_elevations)

        return new_profile


    def clean(self, distance=None):
        """
        :param distance: The length in mm desired between road elevation measurements
        :return: A new immutable RoadProfile class instance
        """
        diffs = np.diff(self.distances)
        inds = np.where(diffs > 0)[0] + 1
        new_dists, new_elevs = self.distances[inds], self.elevations[inds]
        if(len(new_dists) > 0):
            length = new_dists[-1]
            if distance is None:
                min_dist = min(np.min(np.diff(new_dists)), .25) #for now, we are going to interpolate at min distance or 300 mm
                #TODO: Probably need some check in case there are a few distances closely spaced, and others not so
            else:
                min_dist = min(distance, length)
            final_dists = np.linspace(0, length, int(length/min_dist))
            final_elevs = np.interp(final_dists, new_dists, new_elevs)
            return RoadProfile(final_dists, final_elevs)
        return None




    def length(self):
        """

        :return: The length of the entire profile in meters
        """
        to_ret = None
        if(len(self.distances) > 0 ):
            to_ret = self.distances[-1]
        return to_ret

    def split(self, distance):
        """

        :param distance: The distance in meters in which the road profile should be split
        :return: A list of RoadProfile split via distance
        """



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
        # parameters taken from Sayers paper, correspond to Golden Car simluation parameters, more details
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


        x1, smooth_slps = self.compute_smoothed_slopes()
        if(x1 is not None):
            ex_a = expm(a_mat * .25 / vel)  # just use .25 as sample rate
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


