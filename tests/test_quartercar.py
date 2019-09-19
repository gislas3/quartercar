from quartercar import qc
import tests.make_profile as mp
from matplotlib import pyplot as plt
from quartercar import roadprofile as rp
from tests import qc_diff_eq_solver_harmonic as qc_deqs
import numpy as np




def test_sinusodial1(): #TODO - Test varied velocities
    wavelen, amp, prof_len, delta = 1, 10, 100, .01 #2 meter long wavelength, 20 mm amplitude, 100 meters long, spacing of .01 meters bet samples
    velocity = 10 #10 m/s
    distances, heights = mp.make_sinusodal(wavelen, amp, prof_len, delta)
    m_s, m_u, c_s, k_s, k_u = 208, 28, 1300, 18709, 127200 #QC parameters
    qc1 = qc.QC(m_s, m_u, c_s, k_s, k_u)
    rp1 = rp.RoadProfile(distances, heights)
    T, yout, xout = qc1.run(rp1,  100, velocity, 100)
    print("T.shape is {0}, y.shape is {1}, x.shape is {2}".format(T.shape, yout.shape, xout.shape))


    #print()

    y_true =  qc_deqs.harmonic_solver(m_s, m_u, c_s, k_s, k_u, amp/1000, wavelen, velocity, T)

    #Uncomment the following 3 lines to view the plot
    #plt.plot(T, yout[:, -1], color='r')
    #plt.plot(T, y_true, color='g')
    #plt.show()


    assert(np.allclose(yout[:, -1][100:], y_true[100:], .05)) #need to tweak index based on wavelength - longer wavelengths, more time to converge to correct solution
    #print(final_states.shape())
    #Solving for the acceleration of the sprung mass

def test_sinusodial2():
    pass
