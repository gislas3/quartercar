
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


    def clean(self, distance):
        """
        :param distance: The length in mm desired between road elevation measurements
        :return: A new immutable RoadProfile class instance
        """
        pass

    def length(self):
        """

        :return: The length of the entire profile in meters
        """
        return np.sum(self.distances)

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
        # parameters taken from Sayers paper, correspond to quarter car simluation parameters, more details
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


