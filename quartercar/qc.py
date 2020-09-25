import numpy
from .roadprofile import RoadProfile
import numpy as np
from scipy import signal, integrate
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
import logging

class QC():
    """
    A quarter car model that represents the interplay between a road surface (`RoadProfile`), a fixed (and always grounded) mass and a sprung mass.
    The class
      1) calculates acceleration values from a `RoadProfile` and velocities (see `run`) and
      2) reverses the calculation to generate a `RoadProfile` from accelerations, distances and velocities.
    """

    def __init__(self, m_s=None, m_u=None, c_s=None, k_s=None, k_u=None, epsilon=None, omega_s=None, omega_u=None, xi=None, tire_base=.25): #TODO: Put in all different car parameters (with defaults) as constructor arguments
        """
        :param c_s: The suspension damping rate (assumes units N*s/m)
        :param k_s: The suspension spring constant (assumes units N/m)
        :param k_l: The tire spring constant (assumes units N/m)
        :param m_s: The sprung mass (assumes units kg)
        :param m_u: The unsprung mass (assumes units kg)
        :param epsilon: Defined as m_s/m_u
        :param omega_s: Defined as sqrt(k_s/m_s)
        :param omega_u: Defined as sqrt(k_u/m_u)
        :param xi: Defined as c_s/(2*m_s*omega_s)
        """
        #self.c_s = c_s #suspension damping rate (in N * s/m)
        #self.k_s = k_s #suspension spring constant (in units N/m)
        #self.k_l = k_l #tire spring constant (in units N/m)
        #self.m_s  = m_s #sprung mass (in units kg)
        #self.m_u = m_u #unspring mass (in units kg)
        #TODO: Check to make sure parameters valid
        if m_s is not None:
            self.m_s = m_s
            #divide coefficients by sprung mass to make for easier computation into IRI algorithm
            #Can always get back originals by multiplying by m_s
            self.c = c_s/self.m_s
            self.k1 = k_u/self.m_s
            self.k2 = k_s/self.m_s
            self.mu = m_u/self.m_s
        else:
            if epsilon is None: #Use average values if epsilon is None
                epsilon = 5
                omega_s = 1
                omega_u = 10
                xi = .55
                #Aside: According to page 938 of Vehicle Dynamics, 1st Edition the average and min practical values
                # for the above respective parameters are as follows (IMPORTANT: I believe these ratios are for lbs/inch):
                #epsilon: avg = 3 to 8, min = 2, max = 20
                #omega_s: avg = 1, min = .2, max = 1
                #omega_u: avg = 10, min = 2, max = 20
                #xi: avg = .55, min = 0, max = 2
            self.m_s = None
            self.c = xi * 2 * omega_s  # This is c_s/(2*m_s*omega_s) * 2*omega_s = c_s/m_s
            self.k1 = omega_u ** 2 * 1 / epsilon  # This is sqrt(k_u/m_u)**2 * 1/(m_s/m_u) = k_u/m_s
            self.k2 = omega_s ** 2  # This is sqrt(k_s/m_s)**2 = k_s/m_s
            self.mu = 1 / epsilon  # This is 1/(m_s/m_u) = m_u/m_s
        self.tire_base_len = tire_base
        

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


    def run(self, road_profile, distances, velocities, sample_rate_hz=100):
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

        #print("A shape is {0}".format(a.shape))
        b = np.array([[0], [0], [0], [self.k1 / self.mu]])
        #print("B shape is {0}".format(b.shape))
        c = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [self.k2 / self.mu, self.c / self.mu, -(self.k1 + self.k2) / self.mu, -self.c / self.mu], [-self.k2, -self.c, self.k2, self.c]])
        #print("C shape is {0}".format(c.shape))
        #More information about the state space can be found in http://onlinepubs.trb.org/Onlinepubs/trr/1995/1501/1501-001.pdf
        #Basically, we just want dx/dt = Ax + bu, y = x
        state_space = signal.StateSpace(a, b, c, None)
        # for now, we are going to use distance instead of time, and we are going to assume that it's already evenly spaced
        # otherwise, do something like below
        #rp_cleaned = road_profile.clean()
        #TODO: Appropriate error checking below

        # Get the points touched by the car at the given velocity, distance car traveled at velocity, and sample rate
        road_sample = road_profile.car_sample(distances, velocities, sample_rate_hz)
        #print("road sample elevations 0 is {0}".format(road_sample.get_elevations()[0]))
        #road_sample = road_sample.moving_avg_filter()
        if isinstance(velocities, (int, float)) and isinstance(distances, (int, float)):
            #should already be spaced evenly
            velocity = velocities
        else:
            dx = np.average(np.diff(road_sample.get_distances())) #evenly space, using linear interpolation/average difference between points
            total_time = np.sum(distances / velocities)
            road_sample = road_sample.space_evenly(dx) #resample with even spacing in between consecutive samples
            velocity = road_sample.length() / total_time #use average
        #print(road_sample)
        #assume always starting from zero
        #times = np.concatenate((np.zeros(1), np.cumsum(np.diff(road_sample.get_distances()) / velocity)))
        times = np.concatenate((np.zeros(1), np.cumsum(np.diff(road_sample.get_distances()) / velocity)))
        #print("Times[0] is {0}".format(times[0]))
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
        #print("Times 0 is {0}, elevations 0 is {1}".format(times[0], road_sample.get_elevations()[0]))
        T, yout, xout = signal.lsim(state_space, road_sample.get_elevations()/1000, times, x0.reshape(1, -1))

        #print("T.shape is {0}, y.shape is {1}, x.shape is {2}".format(T.shape, yout.shape, xout.shape))

        return T, yout, xout, road_sample.get_distances(), road_sample.get_elevations()

    def next_state(self, t0, p0, v0, delta_t, acc):
        """
        next_state() will return the next time, position, and velocity of the car as it drives along the request road profile.
        :param t0: The initial time (in units seconds) of the simulation
        :param p0: The initial instantaneous position (in units meters) of the car along the road profile 
        :param v0: The initial instantaneous velocity (in units m/s) of the car at the time of the simulation
        :param delta_t: The amount of time the car will be accelerating (in units seconds)
        :param acc: The acceleration (in units m/s**2) that the car will undergo during delta_t (assumes constant acceleration)
        :return: t0 (the next time), p0 (the next instantaneous position), v0 (the next isntantaneous velocity)
        """
        # we aren't going to let the car go backwards, check for that here
        #if v0 == 0 and acc < 0:
        #    acc = abs(acc) #change to positive acceleration so we don't have a bunch of useless data points
        delta_p = delta_t * v0 + .5 * delta_t ** 2 * acc
        delta_v = acc * delta_t
        if delta_p < 0:
            delta_p = 0
        #if delta_v < 0:
        #    delta_v = 0
        t0 += delta_t
        p0 = p0 + delta_p
        v0 = max(v0 + delta_v, 0) #can't have negative velocity, that would be very bad
        return t0, p0, v0


    def run2(self, road_profile,  accelerations, v0=0, delta_t=1e-3, final_sample_rate=None):
        """
        run2 will return a sampled acceleration series of the given road profile, where each
        point corresponds to the acceleration of the qc model at a given time. As of now, doesn't
        take into account any vehicle dynamics (roll/pitch) aside from the vertical vibration
        :param road_profile: The `RoadProfile` object that defines the road to perform the simulation on
        :param accelerations: Either a list of tuples, where each tuple defines how long (in time) to perform the
        acceleration for (so it should be a list like: [(acc1, time1), (acc2, time2), ... (accn, timen)] or a tuple where
        the first element is a scipy random object (defines the distribution of accelerations to pull from) and the second
        is a number type that defines the amount of time to perform the simulation for
        :param v0: The initial velocity (units m/s) default: 0
        :param delta_t: The spacing in time (in seconds) over which to discretize the simulation/differential equation
        :param final_sample_rate: The final sample rate of the series to return. If none, returns the series sampled at delta_t.
        :return: list of acceleration values (in m/s^2)
        """
        #curr_dist = 0
        t0, p0 = 0, 0
        input_times = [0]
        input_elevations = [road_profile.get_next_sample(0)]
        distances = [0]
        velocities = [v0]
        if type(accelerations) == list: #if a list input, this means accelerations pre defined
            for tup in accelerations:
                acc, time = tup[0], tup[1]
                ref_t = 0
                #If you don't calculate the times/velocity/accelerations properly, will just return up to the end of the profile
                while ref_t < time and p0 < road_profile.length():
                    t0, p0, v0 = self.next_state(t0, p0, v0, delta_t, acc)
                    #print("t0 is {0}, v0 is {1}, p0 is {2}".format(t0, v0, p0))
                    elev = road_profile.get_next_sample(p0, base_len=self.tire_base_len)
                    input_times.append(t0)
                    input_elevations.append(elev)
                    velocities.append(v0)
                    ref_t += delta_t
                    #curr_dist += p0
                    distances.append(p0)
                #print("P0 is {0}, v0 is {1}".format(p0, v0))
            while p0 < road_profile.length(): #in case v0 is 0, set a constant acceleration of 1 second (I know this is pretty arbitrary)
                if v0 == 0:
                    acc = 5
                    logging.warning("""v0 is 0, no more accelerations input, and haven't reached end of profile. \n Profile length is {0}, final pos is {1}. 
                    Inputting constant acceleration of 5 for .001 second to rectify """.format(road_profile.length(), p0))
                else:
                    acc = 0
                t0, p0, v0 = self.next_state(t0, p0, v0, delta_t, acc)
                elev = road_profile.get_next_sample(p0, base_len=self.tire_base_len)
                input_times.append(t0)
                input_elevations.append(elev)
                distances.append(p0)
                velocities.append(v0)
                acc = 0
        elif type(accelerations) == tuple: #if a tuple input, this means need to pull accelerations from distribution
            distribution, time = accelerations[0], accelerations[1]
            ref_t = 0
            while p0 < road_profile.length() and t0 < time:
                #t0 += delta_t
                acc = distribution.rvs()
                t0, p0, v0 = self.next_state(t0, p0, v0, delta_t, acc)

                elev = road_profile.get_next_sample(p0, base_len=self.tire_base_len)
                input_times.append(t0)
                input_elevations.append(elev)
                distances.append(p0)
                velocities.append(v0)
        else:
            raise(ValueError("""Unknown acceleration input, please either input a list of tuples where each tuple specifies
                             the acceleration/time or a tuple that specifies the distribution of accelerations/time of experiment"""))

        input_times = np.array(input_times)
        input_elevations = np.array(input_elevations)
        velocities = np.array(velocities)
        distances = np.array(distances)
        a = np.array(
            [[0, 1, 0, 0], [-self.k2, -self.c, self.k2, self.c], [0, 0, 0, 1],
             [self.k2 / self.mu, self.c / self.mu, -(self.k1 + self.k2) / self.mu, -self.c / self.mu]])
        b = np.array([[0], [0], [0], [self.k1 / self.mu]])
        # print("B shape is {0}".format(b.shape))
        c = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                      [self.k2 / self.mu, self.c / self.mu, -(self.k1 + self.k2) / self.mu, -self.c / self.mu],
                      [-self.k2, -self.c, self.k2, self.c]])
        # print("C shape is {0}".format(c.shape))
        # More information about the state space can be found in http://onlinepubs.trb.org/Onlinepubs/trr/1995/1501/1501-001.pdf
        # Basically, we just want dx/dt = Ax + bu, y = x
        state_space = signal.StateSpace(a, b, c, None)
        x0 = np.zeros(4).reshape(4, -1)
        #print("LEN times is {0}, len elevations is {1}".format(len(input_times), len(input_elevations)))
        T, yout, xout = signal.lsim(state_space, input_elevations/1000, input_times, x0.reshape(1, -1))

        if final_sample_rate is not None:
            new_delta_t = 1/final_sample_rate
            if new_delta_t < delta_t: #Can't have a sample rate higher than 1/delta_t
                logging.warning("You are trying to return a sample rate higher than you set the initial sample rate, so this is just going to return the originally sampled series")
                #print("You are trying to return a sample rate higher than you set the initial sample rate, so this is just going to return the originally sampled series")
            elif new_delta_t % delta_t != 0:
                logging.warning("You are trying to downsample at a non-integer downsampling rate - this is not supported (yet), so just returning orignally sampled series")
                #print("You are trying to downsample at a non-integer downsampling rate - this is not supported (yet), so just returning orignally sampled series")
            else:
                factor = int(new_delta_t/delta_t)
                T = T[list(range(0, len(T), factor))]
                yout = yout[list(range(0, len(yout), factor)), ]
                xout = xout[list(range(0, len(xout), factor)), ]
                distances = distances[list(range(0, len(distances), factor))]
                input_elevations = input_elevations[list(range(0, len(input_elevations), factor))]
                velocities = velocities[list(range(0, len(velocities), factor))]

        return T, yout, xout, distances, input_elevations, velocities


        #
        #return accs

    def inverse_time(self, times, accelerations, sample_rate_hz, interp_dt=.01, interp_type='linear', 
            Wn_integrate=None, Wn_elevations=None, f_order=4, space_evenly=False, velocity=0):
        """
        :param times: An array of the measurement times of the acceleration series
        :param accelerations: An array of vertical accelerations in (m/s^2) (assumes that preprocessing step,
            subtracting gravity/accounting for orientation has already occurred)
        :param sample_rate_hz: The sample rate of the series in Hz
        :param interp_dt: The spacing between samples to use if interpolation needs to occur for even sampling, default: .01 (100 Hz sampling rate)
        :param interp_type: Describes the type of interpolation to use between samples (default: 'linear')
        :param Wn_integration: Cutoff frequency to use for high pass filter for drift removal of numerical integration. default: None
        :param Wn_elevations: The cutoff frequency to use for high pass filter for drift removal of final elevations
        :param f_order: The order of filter to use (if needed) for the high pass filters. Default: 4
        :param space_evenly: A boolean representing whether or not to return an evenly spaced road profile. Default: False. If set to True, must 
                            specify velocity > 0 else a ValueError will be raised
        :param velocity: A float representing the velocity to use for returning an evenly spaced road profile. Default: 0
        :returns: input_times: The input times that the accelerations were recorded at
                  input_accs: The input accelerations for estimation of the road profile
                  elevations: The elevations (in m)
        """
        #Step 1: check to see if the samples have constant spacing
        if not np.allclose(np.diff(np.diff(times)), np.zeros(len(times)-2), 1e-4):
            #need to interpolate to space evenly
            input_times = np.arange(times[0], times[-1]+interp_dt, interp_dt)
            if interp_type == 'linear':
                input_accs = np.interp(input_times, times, accelerations)
            elif interp_type == 'cs':
                cs = CubicSpline(input_times, accelerations)
                input_accs = cs(input_times)
            else:
                raise ValueError("An invalid interpolation type was specified. The following interpolation types are currently supported: 'linear', 'cs'")
        else:
            input_times = times
            input_accs = accelerations

        #Step 2: Integrate in the time domain the accelerations
        x_s_dot = integrate.cumtrapz(input_accs, input_times, initial=0)
        if Wn_integrate is not None:
            sos = signal.butter(f_order, Wn_integrate, 'highpass', fs=1/(input_times[1] - input_times[0]), output='sos')
            x_s_dot = signal.sosfilt(sos, x_s_dot)
        x_s = integrate.cumtrapz(x_s_dot, input_times, initial=0)
        if Wn_integrate is not None:
            x_s = signal.sosfilt(sos, x_s)
        #Step 3: Set up state space for predicting motion of unsprung mass
        a = -self.k2/self.c #k2 = k_s/m_s, c = c_s/m_s, -k2/c = -k_s/c_s
        b = 1
        state_space = signal.StateSpace(a, b, a, b)
        U = input_accs*1/self.c + x_s_dot + self.k2/self.c*x_s #y = m_s/c_s*x_dot_dot + c_s*x_dot + k_s/c_s*x_s; c = c_s/m_s, 1/c = m_s/c_s; c_s/c_s = 1; k2 = k_s/m_s, c = c_s/m_s, k2/c = k_s/c_s
        _, x_u_dot, x_u = signal.lsim(state_space, U, input_times)
        
        #Step 4: now, compute numerical derivatives of x_u_dot_dot
        x_u_dot_dot = np.concatenate((np.zeros(1), np.diff(x_u_dot)/np.diff(input_times)))

        #Step 5: now, compute road profile
        elevations = self.mu/self.k1*x_u_dot_dot + self.c/self.k1*(x_u_dot - x_s_dot) + (self.k1+self.k2)/self.k1*x_u - self.k2/self.k1*x_s
        if Wn_elevations is not None:
            sos_el = signal.butter(f_order, Wn_elevations, 'highpass', fs=1/(input_times[1] - input_times[0]), output='sos')
            elevations = signal.sosfilt(sos_el, elevations)
        
        #Step 6: possibly evenly space the elvations
        if space_evenly:
            if velocity is None or velocity < 0:
                logging.warning("Space_evenly was set to True but an invalid velocity was specified, the program is not returning evenly spaced elevations")
            else: #not going to worry about this yet, because not sure it makes sense on second thought
                pass
        return input_times, input_accs, elevations*1000



    def inverse(self, accelerations, distances, velocities, sample_rate_hz=100, interp_dx=None, interp_type=None, Wn = None):

        """


        :param accelerations: An array of vertical accelerations in (m/s^2) (assumes that preprocessing step,
            subtracting gravity/accounting for orientation has already occurred)
        :param distances: An array that contains the distance (in mm) from start of measurements
        :param velocities: An array that contains the instantaneous velocity of the car at each distance
        :param: interp_dx: The parameter specifying what the spacing should be if interpolating the acceleration series
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


        if isinstance(velocities, (int, float)):
            velocity = velocities
            times = np.concatenate((np.zeros(1), np.cumsum(np.diff(distances) / velocity)))
            even_dists = None
        else:
            dist_treshold = 5 #in meters
            if sum(np.diff(distances) > dist_treshold) != 0:
                raise Exception('Spacing of samples is larger than {0} meters'.format(dist_treshold))
            if interp_dx is None:
                #Choose average distance
                delta = np.average(np.diff(distances))
            else:
                delta = interp_dx  # desired space between measurements eg. 250 mm, or 300 mm

                length_of_trip = distances[-1] # the last measurement is the total length of the trip
                number_of_samples = int(length_of_trip / delta) + 1
                even_dists = np.linspace(0, length_of_trip, number_of_samples)
                #print("Even dists is {0}".format(even_dists))
                if interp_type == 'linear':
                    even_acc = np.interp(even_dists, distances, accelerations)

                elif interp_type == 'polynomial':
                    cs = CubicSpline(distances, accelerations)
                    even_acc = cs(even_dists)

                else: #if interp_type is None
                    if delta < 1:
                        cs = CubicSpline(distances, accelerations)
                        even_acc = cs(even_dists)
                    else:
                        even_acc = np.interp(even_dists, distances, accelerations)

                # We choose the smallest speed for simulation
                min_velocity = min(velocities)
                if min_velocity == 0:
                    min_velocity = 10  # in m/s

                times = np.concatenate(
                    (np.zeros(1), np.cumsum(np.diff(even_dists) / min_velocity)))  # times measured from start in seconds
                #return None
                distances = even_dists
                accelerations = even_acc
                velocity = min_velocity

        #Step 1: Numerically integrate x_s_dot_dot to get x_s_dot, then numerically integrate x_s_dot to get x_s
        sample_rate_per_meter = int(sample_rate_hz / velocity)
        if Wn == None:
            longest_wave_len = 91 #longest wavelength of interest
            sos = signal.butter(10, 1 / longest_wave_len, 'highpass', fs=sample_rate_per_meter, output='sos')
        else:
            sos = signal.butter(10, Wn, 'highpass', fs=sample_rate_per_meter, output='sos')

        x_s_dot = integrate.cumtrapz(accelerations, times, initial=0)
        #x_s_dot = signal.sosfilt(sos, x_s_dot)
        x_s = integrate.cumtrapz(x_s_dot, times, initial=0)
        #x_s = signal.sosfilt(sos, x_s)
        #Step 2:
        a = -self.k2/self.c #k2 = k_s/m_s, c = c_s/m_s, -k2/c = -k_s/c_s
        b = 1
        state_space = signal.StateSpace(a, b, a, b)
        U = accelerations*1/self.c + x_s_dot + self.k2/self.c*x_s #y = m_s/c_s*x_dot_dot + c_s*x_dot + k_s/c_s*x_s; c = c_s/m_s, 1/c = m_s/c_s; c_s/c_s = 1; k2 = k_s/m_s, c = c_s/m_s, k2/c = k_s/c_s
        _, x_u_dot, x_u = signal.lsim(state_space, U, times)
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
        if Wn == None:
            elevations_filt = elevations
        else:
            elevations_filt = signal.sosfilt(sos, elevations)

        return elevations_filt*1000, x_s_dot, x_s, x_u, x_u_dot, x_u_dot_dot, distances, accelerations





        #return road_profile
        #pass

    def transfer_function(self, frequencies, psd, veloc):
        """
        Method for computing the transfer function for this car's parameters given road input frequencies, road
        power spectral density, and velocity
        :param frequencies: Frequencies of the road power spectral density, assumed to be in units cycles/meter
        :param psd: Power spectral density of the road profile the car will be driven over
        :param veloc: The velocity that the car will be driving over the profile at, should in in units m/s
        :return: time_frequencies (the new frequencies converted to the time domain, in units Hz/cycles/second)
                acc_psd (the acceleration power spectral density using the transfer function)
        """
        ku = self.k1 * self.m_s
        ks = self.k2 * self.m_s
        ms = self.m_s
        mu = self.mu * self.m_s
        cs = self.c * self.m_s
        # new_freqs = orig_freqs * veloc * 2 * np.pi
        time_frequencies = frequencies * veloc * 2 * np.pi #convert to angular frequency for input to transfer function
        # t_func = new_freqs**2 * (cs*ku*new_freqs + ks*ku)/(mu*ms*new_freqs**4 + (cs*mu + cs*ms)*new_freqs**3 +
        #                                                   (ms*ks+mu*ks+ms*ku)*new_freqs**2 + (cs*ku)*new_freqs + ks*ku)
        transfer_func = (time_frequencies ** 2 * np.sqrt(cs ** 2 * ku ** 2 * time_frequencies ** 2 + ks ** 2 * ku ** 2)) / np.sqrt(
            (cs * ku * time_frequencies + time_frequencies ** 3 * (-cs * ms - cs * mu)) ** 2 + (
                        time_frequencies ** 2 * (-ks * ms - ks * mu - ku * ms)
                        + ks * ku + ms * mu * time_frequencies ** 4) ** 2)
        return time_frequencies / (2 * np.pi), transfer_func ** 2 * (psd/veloc)


    def inverse_transfer_function(self, frequencies, psd, veloc):
        """
                Method for computing the inverse transfer function for this car's parameters given road input frequencies, road
                power spectral density, and velocity
                :param frequencies: Frequencies of the acceleration power spectral density, assumed to be in units cycles/meter
                :param psd: Acceleration power spectral density
                :param veloc: The velocity that the car drover over the profile at, should be in units m/s
                :return: spatial_frequencies (the new frequencies converted to the spatial domain, in units cycles/meter)
                        road_psd (the road power spectral density using the transfer function)
        """
        ku = self.k1 * self.m_s
        ks = self.k2 * self.m_s
        ms = self.m_s
        mu = self.mu * self.m_s
        cs = self.c * self.m_s
        frequencies = frequencies * 2* np.pi #need to convert to angular frequency
        transfer_func = (frequencies ** 2 * np.sqrt(
            cs ** 2 * ku ** 2 * frequencies ** 2 + ks ** 2 * ku ** 2)) / np.sqrt(
            (cs * ku * frequencies + frequencies ** 3 * (-cs * ms - cs * mu)) ** 2 + (
                    frequencies ** 2 * (-ks * ms - ks * mu - ku * ms)
                    + ks * ku + ms * mu * frequencies ** 4) ** 2)
        road_psd = (psd * veloc)/(transfer_func)**2
        road_psd[np.where(road_psd == np.inf)] = 0
        road_psd[np.where(np.isnan(road_psd))] = 0
        #road_psd[np.where(road_psd == np.nan)] = 0
        #road_psd[np.where(road_psd == np.nan)] = 0
        spatial_frequencies = frequencies/(2*np.pi * veloc)
        return spatial_frequencies, road_psd