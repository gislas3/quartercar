import numpy
from .roadprofile import RoadProfile
import numpy as np
from scipy import signal, integrate
from matplotlib import pyplot as plt

class QC():
    """
    A quarter car model that represents the interplay between a road surface (`RoadProfile`), a fixed (and always grounded) mass and a sprung mass.
    The class 
      1) calculates acceleration values from a `RoadProfile` and velocities (see `run`) and 
      2) reverses the calculation to generate a `RoadProfile` from accelerations, distances and velocities.
    """

    def __init__(self, m_s=0, m_u=0, c_s=0, k_s=0, k_u=0): #TODO: Put in all different car parameters (with defaults) as constructor arguments
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
        self.k1 = k_u/self.m_s
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
            x0 = np.array([[init_val/1000], [0], [init_val/1000], [0]])
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






    def run(self, road_profile,  distances, velocities, sample_rate_hz=100):
        """
        This is where we generate the acceleration values from the road profile, running the entire QC simulation

        :param road_profile: An array or list like of floating point values (in mm) of road profile elevations
        :param velocities: Either an array or an float. If an array is given it is assumed to contain the instantaneous velocity of the car over specified distances (in m/s). If a float is given, it is assumed to be a fixed velocity for all road profile points.
        :param distances: Either an array or an float. If an array is given it is assumed to contain a one-to-one mapping of distance traveled at each specified velocity in velocities (in m). If a float is given the data points are assumed to be evenly spaced with the given distance.
        :param sample_rate_hz: The sampling rate of the accelerometer attached to the vehicle body in Hz (default: 100 Hz)
        :return: list of acceleration values (in m/s^2)
        """
        #sample_rate_hz = 100
        a = np.array(
            [[0, 1, 0, 0], [-self.k2, -self.c, self.k2, self.c], [0, 0, 0, 1], [self.k2 / self.mu, self.c / self.mu, -(self.k1 + self.k2) / self.mu, -self.c / self.mu]])

        print("A shape is {0}".format(a.shape))
        b = np.array([[0], [0], [0], [self.k1 / self.mu]])
        print("B shape is {0}".format(b.shape))
        c = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-self.k2, -self.c, self.k2, self.c]])
        print("C shape is {0}".format(c.shape))
        #More information about the state space can be found in http://onlinepubs.trb.org/Onlinepubs/trr/1995/1501/1501-001.pdf
        #Basically, we just want dx/dt = Ax + bu, y = x
        state_space = signal.StateSpace(a, b, c, None)
        # for now, we are going to use distance instead of time, and we are going to assume that it's already evenly spaced
        # otherwise, do something like below
        #rp_cleaned = road_profile.clean()
        #TODO: Appropriate error checking below

        # Get the points touched by the car at the given velocity, distance car traveled at velocity, and sample rate
        road_sample = road_profile.car_sample( distances,velocities,sample_rate_hz)
        if isinstance(velocities, (int, float)) and isinstance(distances, (int, float)):
            #should already be spaced evenly
            velocity = velocities
        else:
            dx = np.average(np.diff(road_sample.get_distances())) #evenly space, using linear interpolation/average difference between points
            total_time = np.sum(distances / velocities)
            road_sample = road_sample.space_evenly(dx) #resample with even spacing in between consecutive samples
            velocity = road_sample.length() / total_time #use average
        print(road_sample)
        #assume always starting from zero
        #times = np.concatenate((np.zeros(1), np.cumsum(np.diff(road_sample.get_distances()) / velocity)))
        times = np.concatenate((np.zeros(1), np.cumsum(np.diff(road_sample.get_distances()) / velocity)))
        print("Times[0] is {0}".format(times[0]))
        #if using average dx:

        #if using min dx:
        #dx = np.min(np.diff(road_sample.get_distances()))



        # should be even since road profile sampled evenly
        
        #plt.plot(times, road_sample.get_elevations()/1000)
        #plt.xticks(np.arange(0, 10, .1), [0] + 10*[None] + [1] + 10*[None] + [2] + 77*[None])
        #plt.vlines(.2, -.02, .02)
        #plt.show()
        #x0 = self.get_initial_state(road_sample)
        x0 = np.zeros(4).reshape(4, -1) #Didn't seem to have much effect when looking at the plots how we initialized the system
        #print("len(road_sample.get_elevations()) is {0}, len(times) is {1}".format(len(road_sample.get_elevations()), len(times)))
        T, yout, xout = signal.lsim(state_space, road_sample.get_elevations()/1000, times, x0.reshape(1, -1))

        #print("T.shape is {0}, y.shape is {1}, x.shape is {2}".format(T.shape, yout.shape, xout.shape))

        return T, yout, xout, road_sample.get_distances(), road_sample.get_elevations()





        #
        #return accs




    def inverse(self, accelerations, distances, velocities, sample_rate_hz=100):

        """

        TODO: Still need to figure out the best way to do this
        :param accelerations: An array of vertical accelerations in (m/s^2) (assumes that preprocessing step,
            subtracting gravity/accounting for orientation has already occurred)
        :param distances: An array that contains the distance (in mm) from start of measurements
        :param velocities: An array that contains the instantaneous velocity of the car at each distance

        :return: `RoadProfile`
        """

        # I think for now the only thing we could really do is:
        #1. Numerically integrate x_s_dot_dot to solve for x_s_dot and x_s
        #2. Set up a state space using the first equation for x_s_dot_dot to solve for x_u and x_u_dot
        # So for instance, rewrite: m_s*x_s_dot_dot + c_s(x_s_dot - x_u_dot) + k_s(x_s - x_u) = 0
        # as: x_u_dot = (-k_s*x_u + y)/c_s, where y = m_s*x_s_dot_dot + c_s*x_s_dot + ks_*x_s
        # then, estimate initial condition and solve for x_u and x_u_dot at each time step
        #3. Estimate x_u_dot_dot from the time derivative of x_u_dot
        #4. Use estimated values of x_s_dot, x_s, x_u_dot_dot, x_u_dot, and x_u to determine y
        #E.G. plug in values at each time step to find:
        # (m_u*x_u_dot_dot + c_s(x_u_dot - x_s_dot) + (k_u + k_s)*x_u - k_s*x_s)/k_u = y

        #TODO: More research to see if there is a more elegant/better way to solve for the road profile (i.e. generate data and look at ways to go back to it)
        if isinstance(velocities, (int, float)):
            velocity = velocities
            times = np.concatenate((np.zeros(1), np.cumsum(np.diff(distances) / velocity)))
        else: #TODO: Figure out way to evenly space/interpolate between acceleration values

            return None

        #Step 1: Numerically integrate x_s_dot_dot to get x_s_dot, then numerically integrate x_s_dot to get x_s
        sample_rate_per_meter = int(sample_rate_hz / velocity)
        longest_wave_len = 91 #longest wavelength of interest

        sos = signal.butter(10, 1 / longest_wave_len, 'highpass', fs=sample_rate_per_meter, output='sos')
        x_s_dot = integrate.cumtrapz(accelerations, times, initial=0)
        #x_s_dot = signal.sosfilt(sos, x_s_dot)
        x_s = integrate.cumtrapz(x_s_dot, times, initial=0)
        #x_s = signal.sosfilt(sos, x_s)
        #Step 2:
        a = -self.k2/self.c #k2 = k_s/m_s, c = c_s/m_s, -k2/c = -k_s/c_s
        b = 1
        state_space = signal.StateSpace(a, b, a, b)
        U = accelerations*1/self.c + x_s_dot + self.k2/self.c*x_s #y = m_s/c_s*x_dot_dot + c_s*x_dot + k_s/c_s*x_s; c = c_s/m_s, 1/c = m_s/c_s; c_s/c_s = 1; k2 = k_s/m_s, c = c_s/m_s, k2/c = k_s/c_s
        T, x_u_dot, x_u = signal.lsim(state_space, U, times)
        #print("X_u shape is {0}".format(x_u[2].shape))
        #x_u_dot = a*x_u + U

        #Step 3:

        x_u_dot_dot = np.concatenate((np.zeros(1), np.diff(x_u_dot)/np.diff(times)))

        #Step 4:

        #(m_u*x_u_dot_dot + c_s(x_u_dot - x_s_dot) + (k_u + k_s)*x_u - k_s*x_s)/k_u = y
        #mu = m_u/m_s, k1 = k_u/m_s, mu/k1 = m_u/k_u; c = c_s/m_s, k1 = k_u/m_s, c/k1 = c_s/k_u; k1 = k_u/m_s, k2 = k_s/m_s, (k1 + k2)/k1 = (k_u + k_s)/k_u; -k2 = -k_s/m_s, k1 = k_u/m_s, -k2/k1 = -k_s/k_u;
        elevations = self.mu/self.k1*x_u_dot_dot + self.c/self.k1*(x_u_dot - x_s_dot) + (self.k1+self.k2)/self.k1*x_u - self.k2/self.k1*x_s


        #plt.plot(times, elevations)
        #let's try just filtering the profile at the end, with a high pass filter (in the spatial domain) with a cutoff of 91 meters

        elevations_filt = signal.sosfilt(sos, elevations)
        return elevations_filt*1000, x_s_dot, x_s, x_u, x_u_dot, x_u_dot_dot





        #return road_profile
        #pass






