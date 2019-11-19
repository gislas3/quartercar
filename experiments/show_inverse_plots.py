from quartercar import qc
import tests.make_profile as mp
from matplotlib import pyplot as plt
from quartercar import roadprofile as rp
from scipy.fftpack import fft
from scipy import signal
from tests import qc_diff_eq_solver_harmonic as qc_deqs
import numpy as np
import pandas as pd


def inverse1():
    wavelen, amp, prof_len, delta = 5, 10, 100, .0001  # 2 meter long wavelength, 20 mm amplitude, 100 meters long, spacing of .01 meters bet samples
    velocity = 10 # 10 m/s
    distances, heights = mp.make_sinusodal(wavelen, amp, prof_len, delta)
    sample_rate_hz = 100
    m_s, m_u, c_s, k_s, k_u = 208, 28, 1300, 18709, 127200  # QC parameters
    qc1 = qc.QC(m_s, m_u, c_s, k_s, k_u)
    rp1 = rp.RoadProfile(distances, heights)
    #plt.plot(rp1.get_distances(), rp1.get_elevations(), color='b')
    #plt.show()
    T, yout, xout, new_distances, new_elevations = qc1.run(rp1, 100, velocity, sample_rate_hz)
    acc_true = qc_deqs.harmonic_solver(m_s, m_u, c_s, k_s, k_u, amp / 1000, wavelen, velocity, T)
    #plt.plot(T, acc_true, color='g')
    #plt.show()

    est_profile, est_xs_dot, est_xs, est_xu, est_xu_dot, est_xu_dot_dot = qc1.inverse(acc_true, new_distances, velocity, sample_rate_hz)

    make_plots = True
    make_fft = False
    use_dist = False
    if make_plots:
        if make_fft:
            N = len(T)
            #total_time = T[-1]
            if use_dist: #then use distance instead of time domain
                sample_rate_per_meter = int(sample_rate_hz / velocity)
                max_freq = int(sample_rate_per_meter / 2) #this is max frequency in spatial domain (frequency in spatial domain is cycles per meter)

                sos = signal.butter(10, 1/90, 'highpass', fs=sample_rate_per_meter, output='sos')
            else:
                max_freq = int(sample_rate_hz / 2) #this is max frequency in time domain
                sos = signal.butter(10, 1/(90/velocity), 'highpass', fs=sample_rate_hz, output='sos') #corresponds to 30 m wavelength highpass filter in spatial domain

            x_f = np.linspace(0.0, max_freq, N//2)
            print("x_f goes from {0} to {1}".format(x_f[0], x_f[-1]))
            x_fft1 = fft(xout[:, 0])
            x_fft2 = fft(est_xs)
            plt.plot(x_f[1:], 2.0 / N * np.abs(x_fft1[1:N // 2]), color='g')
            plt.plot(x_f[1:], 2.0 / N * np.abs(x_fft2[1:N // 2]), color='r')
            #x_filt = signal.sosfilt(sos, est_xs)
            #x_fft3 = fft(x_filt)
            #plt.plot(x_f[1:], 2.0 / N * np.abs(x_fft3[1:N // 2]), color='b')
            plt.title("FFT of x_s true vs. x_s estimated")
            plt.show()

            x_fft1 = fft(xout[:, 1])
            x_fft2 = fft(est_xs_dot)
            plt.plot(x_f[1:], 2.0 / N * np.abs(x_fft1[1:N // 2]), color='g')
            plt.plot(x_f[1:], 2.0 / N * np.abs(x_fft2[1:N // 2]), color='r')
            x_filt = signal.sosfilt(sos, est_xs_dot)
            x_fft3 = fft(x_filt)
            plt.plot(x_f[1:], 2.0 / N * np.abs(x_fft3[1:N // 2]), color='b')
            plt.title("FFT of x_s_dot true vs. x_s_dot estimated vs. x_s_dot filtered")
            plt.show()

            x_fft1 = fft(xout[:, 2])
            x_fft2 = fft(est_xu)
            plt.plot(x_f[1:], 2.0 / N * np.abs(x_fft1[1:N // 2]), color='g')
            plt.plot(x_f[1:], 2.0 / N * np.abs(x_fft2[1:N // 2]), color='r')
            plt.title("FFT of x_u true vs. x_u estimated")
            plt.show()

            x_fft1 = fft(xout[:, 3])
            x_fft2 = fft(est_xu_dot)
            plt.plot(x_f[1:], 2.0 / N * np.abs(x_fft1[1:N // 2]), color='g')
            plt.plot(x_f[1:], 2.0 / N * np.abs(x_fft2[1:N // 2]), color='r')
            plt.title("FFT of x_u_dot true vs. x_u_dot estimated")
            plt.show()

            x_fft1 = fft(new_elevations)
            x_fft2 = fft(est_profile)
            plt.plot(x_f[1:], 2.0 / N * np.abs(x_fft1[1:N // 2]), color='g')
            plt.plot(x_f[1:], 2.0 / N * np.abs(x_fft2[1:N // 2]), color='r')
            x_filt = signal.sosfilt(sos, est_profile)
            x_fft3 = fft(x_filt)
            plt.plot(x_f[1:], 2.0 / N * np.abs(x_fft3[1:N // 2]), color='b')
            plt.title("FFT of True profile vs. Estimated Profile vs Estimated Profile filtered")
            plt.show()



        else:
            plt.plot(T, acc_true, color='g')
            plt.plot(T, yout[:, -1], color='r')
            plt.title("Estimated X_S_Dot_Dot and True X_S_Dot_Dot")
            plt.show()

            plt.plot(T, acc_true, color='g')
            plt.plot(T, est_xs_dot, color='r')
            plt.title("Acceleration vs. Its Numerical Integration Estimation")
            plt.ylim(-10, 10)
            plt.show()

            #plt.plot(yout[:, -1], color='r')

            #plt.plot(T, yout[:, 1], color='g')
            #plt.plot(T, est_xs_dot, color='r')
            #plt.title("Estimated X_S_Dot and True X_S_Dot")
            #plt.show()

            plt.plot(T, xout[:, 0], color='g')
            plt.plot(T, est_xs, color='r')
            plt.title("Estimated X_S and True X_S")
            plt.show()

            plt.plot(T, xout[:, 1], color='g')
            plt.plot(T, est_xs_dot, color='r')
            plt.title("Estimated X_S_Dot and True X_S_Dot")
            plt.show()

            plt.plot(T, xout[:, 2], color='g')
            plt.plot(T, est_xu, color='r')
            plt.title("Estimated X_U and True X_U")
            plt.show()

            plt.plot(T, xout[:, 3], color='g')
            plt.plot(T, est_xu_dot, color='r')
            plt.title("Estimated X_U_Dot and True X_U_Dot")
            plt.show()

            plt.plot(T, new_elevations, color='g')
            plt.plot(T, est_profile, color='r')
            plt.ylim(-100, 100)
            plt.title("Estimated Profile vs. True Profile")
            plt.show()



    #assert(False)

def run_gaussian():
    sigma = 8  # 8 mm
    profile_len, delta, cutoff_freq, delta2, seed = 100, .1, .15, .01, 55
    dists, orig_hts, low_pass_hts, final_dists, final_heights = mp.make_gaussian(sigma, profile_len, delta,
                                                                                cutoff_freq, delta2, seed)


    rp1 = rp.RoadProfile(final_dists, final_heights)
    velocity = 10  # 10 m/s
    m_s, m_u, c_s, k_s, k_u = 208, 28, 1300, 18709, 127200  # QC parameters
    qc1 = qc.QC(m_s, m_u, c_s, k_s, k_u)
    T, yout, xout, new_distances, new_elevations = qc1.run(rp1, 100, velocity, 100)
    plt.plot(T, yout[:, -1])
    plt.title("Sprung mass acceleration from gaussian input")
    plt.show()

    #assert(False)

def test_inverse_gaussian():
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
    est_profile, est_xs, est_xs_dot, est_xu, est_xu_dot, est_xu_dot_dot = qc1.inverse(yout[:, -1], new_distances, velocity,
                                                                                      sample_rate_hz)


    iri_full = rp1.to_iri()
    rp_est = rp.RoadProfile(new_distances, est_profile)
    iri_orig = rp.RoadProfile(new_distances, new_elevations).to_iri()
    iri_est = rp_est.to_iri()
    plt.plot(new_distances, new_elevations, color='g')
    plt.plot(new_distances, est_profile, color='r')
    #data1 = {"distances": new_distances, "elevations": new_elevations}
    #df1 = pd.DataFrame(data1)
    #data2 = {"distances": new_distances, "elevations": est_profile}
    #df2 = pd.DataFrame(data2)
    #df1.to_csv("/Users/gregoryislas/Documents/Mobilized/sampled_profile", index=False)
    #df2.to_csv("/Users/gregoryislas/Documents/Mobilized/estimated_profile", index=False)
    plt.title("Estimated profile (r) (IRI={1}) and sampled input (IRI={2}, real={3}) (g), SR = {0}HZ".format(sample_rate_hz, np.round(iri_est, 3), np.round(iri_orig, 3), np.round(iri_full, 3)))
    plt.show()



    #plt.plot(T, yout[:, -1])
    #plt.title("Sprung mass acceleration from gaussian input")
    #plt.show()

    #assert(False)

#if __name__ == 'main':
print("NAME IS MAIN")
inverse1()
run_gaussian()
test_inverse_gaussian()