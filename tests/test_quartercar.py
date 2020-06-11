from quartercar import qc
import tests.make_profile as mp
from quartercar import roadprofile as rp
from tests import qc_diff_eq_solver_harmonic as qc_deqs
import numpy as np




#Tests that the run function works as expected when the road profile input is a sinusodial
def test_sinusodial1(): #TODO - Test varied velocities
    wavelen, amp, prof_len, delta = 1, 10, 100, .01 #2 meter long wavelength, 20 mm amplitude, 100 meters long, spacing of .01 meters bet samples
    velocity = 10 #10 m/s
    distances, heights = mp.make_sinusodal(wavelen, amp, prof_len, delta)
    m_s, m_u, c_s, k_s, k_u = 208, 28, 1300, 18709, 127200 #QC parameters
    qc1 = qc.QC(m_s, m_u, c_s, k_s, k_u)
    rp1 = rp.RoadProfile(distances, heights)
    T, yout, xout, new_distances, new_elevations = qc1.run(rp1,  100, velocity, 100)
    #print("T.shape is {0}, y.shape is {1}, x.shape is {2}".format(T.shape, yout.shape, xout.shape))


    #print()

    y_true = qc_deqs.harmonic_solver(m_s, m_u, c_s, k_s, k_u, amp/1000, wavelen, velocity, T)

    #Uncomment the following 3 lines to view the plot
    #plt.plot(T, yout[:, -1], color='r')
    #plt.plot(T, y_true, color='g')
    #plt.show()


    assert(np.allclose(yout[:, -1][100:], y_true[100:], .05)) #need to tweak index based on wavelength - longer wavelengths, more time to converge to correct solution
    #print(final_states.shape())
    #Solving for the acceleration of the sprung mass
#Tests that run works as expected using the epsilon, xi, and omega parameters of the constructor
def test_sinusodial2():
    wavelen, amp, prof_len, delta = 1, 10, 100, .01 #2 meter long wavelength, 20 mm amplitude, 100 meters long, spacing of .01 meters bet samples
    velocity = 10 #10 m/s
    distances, heights = mp.make_sinusodal(wavelen, amp, prof_len, delta)
    m_s, m_u, c_s, k_s, k_u = 208, 28, 1300, 18709, 127200 #QC parameters
    eps, w_s, w_u = m_s/m_u, np.sqrt(k_s/m_s), np.sqrt(k_u/m_u)
    xi = c_s/(2*m_s*w_s)
    qc1 = qc.QC(epsilon=eps, omega_s=w_s, omega_u=w_u, xi=xi)
    rp1 = rp.RoadProfile(distances, heights)
    T, yout, xout, new_distances, new_elevations = qc1.run(rp1,  100, velocity, 100)
    #print("T.shape is {0}, y.shape is {1}, x.shape is {2}".format(T.shape, yout.shape, xout.shape))


    #print()

    y_true =  qc_deqs.harmonic_solver(m_s, m_u, c_s, k_s, k_u, amp/1000, wavelen, velocity, T)

    #Uncomment the following 3 lines to view the plot
    #plt.plot(T, yout[:, -1], color='r')
    #plt.plot(T, y_true, color='g')
    #plt.show()


    assert(np.allclose(yout[:, -1][100:], y_true[100:], .05)) #need to tweak index based on wavelength - longer wavelengths, more time to converge to correct solution
    #print(final_states.shape())
    #Solving for the acceleration of the sprung mass

#Tests that the inverse works properly... that it at least returns the length we're expecting
def test_inverse():
    sigma = 8  # 8 mm
    profile_len, delta, cutoff_freq, delta2, seed = 100, .1, .15, .01, 55
    dists, orig_hts, low_pass_hts, final_dists, final_heights = mp.make_gaussian(sigma, profile_len, delta,
                                                                                cutoff_freq, delta2, seed)


    rp1 = rp.RoadProfile(final_dists, final_heights)
    velocity = 10  # 10 m/s
    m_s, m_u, c_s, k_s, k_u = 208, 28, 1300, 18709, 127200  # QC parameters
    qc1 = qc.QC(m_s, m_u, c_s, k_s, k_u)
    sample_rate_hz = 500
    T, yout, xout, new_distances, new_elevations = qc1.run(rp1, 100, velocity, sample_rate_hz)
    est_profile, est_xs, est_xs_dot, est_xu, est_xu_dot, est_xu_dot_dot, acc = qc1.inverse(yout[:, -1], new_distances, velocity,
                                                                                      sample_rate_hz)
    assert(len(est_profile) == len(yout[:, -1]))
#
#

# wavelen, amp, prof_len, delta = 5, 10, 100, .0001  # 2 meter long wavelength, 20 mm amplitude, 100 meters long, spacing of .01 meters bet samples
#    velocity = 10 # 10 m/s
#    distances, heights = mp.make_sinusodal(wavelen, amp, prof_len, delta)

def test_run2_constantacc():
    #Test to make sure it works with one constant acceleration
    distances, elevations = mp.make_profile_from_psd('B', 'sine', .1, 100, seed=69)
    profile = rp.RoadProfile(distances, elevations)
    v0 = 0
    vehicle = qc.QC(m_s=243, m_u=40, c_s=370, k_s=14671, k_u=124660)
    packed_vals = vehicle.run2(profile, [(2, 5)], v0=v0)
    assert(len(packed_vals[0]) == int(12.5*1000) + 1)

def test_run2_constantacc2():
    #test to make sure it works with two constant accelerations
    distances, elevations = mp.make_profile_from_psd('B', 'sine', .1, 100, seed=69)
    profile = rp.RoadProfile(distances, elevations)
    v0 = 0
    vehicle = qc.QC(m_s=243, m_u=40, c_s=370, k_s=14671, k_u=124660)
    packed_vals = vehicle.run2(profile, [(2, 5), (5/2, 2)], v0=v0)
    assert(len(packed_vals[0]) == int((10 + 1/3) *1000) + 1)

def test_run2_constantacc3():
    #Test make sure it works with non constant sample rate
    distances, elevations = mp.make_profile_from_psd('B', 'sine', .1, 100, seed=69)
    profile = rp.RoadProfile(distances, elevations)
    v0 = 0
    vehicle = qc.QC(m_s=243, m_u=40, c_s=370, k_s=14671, k_u=124660)
    packed_vals = vehicle.run2(profile, [(2, 5)], v0=v0, final_sample_rate=100)
    assert (len(packed_vals[0]) == int(12.5 * 1000)/10 + 1)

def test_run2_constantacc4():
    #Test make sure it runs with a decceleration as well
    distances, elevations = mp.make_profile_from_psd('B', 'sine', .1, 100, seed=69)
    profile = rp.RoadProfile(distances, elevations)
    v0 = 0
    vehicle = qc.QC(m_s=243, m_u=40, c_s=370, k_s=14671, k_u=124660)
    packed_vals = vehicle.run2(profile, [(2, 5), (5 / 2, 2), (-5/2, 2)], v0=v0)
    assert (len(packed_vals[0]) == int(11.5 * 1000))


