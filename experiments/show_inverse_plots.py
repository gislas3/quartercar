from quartercar import qc
import tests.make_profile as mp
from matplotlib import pyplot as plt
from quartercar import roadprofile as rp
from scipy.fftpack import fft
from scipy import signal
from tests import qc_diff_eq_solver_harmonic as qc_deqs
import numpy as np
from tests import qc_diff_eq_solver_harmonic as qdeq
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

    est_profile, est_xs_dot, est_xs, est_xu, est_xu_dot, est_xu_dot_dot, acc = qc1.inverse(acc_true, new_distances, velocity, sample_rate_hz)

    make_plots = False
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
    #sigma = 8  # 8 mm
    #profile_len, delta, cutoff_freq, delta2, seed = 100, .1, .15, .01, 55
    #dists, orig_hts, low_pass_hts, final_dists, final_heights = mp.make_gaussian(sigma, profile_len, delta,
    #                                                                            cutoff_freq, delta2, seed)

    final_dists, final_heights = mp.make_profile_from_psd('A', 'sine', .1, 100)
    rp1 = rp.RoadProfile(final_dists, final_heights)
    velocity = 10  # 10 m/s
    m_s, m_u, c_s, k_s, k_u = 208, 28, 1300, 18709, 127200  # QC parameters
    qc1 = qc.QC(m_s, m_u, c_s, k_s, k_u)
    sample_rate_hz = 500
    T, yout, xout, new_distances, new_elevations = qc1.run(rp1, 100, velocity, sample_rate_hz)
    est_profile, est_xs, est_xs_dot, est_xu, est_xu_dot, est_xu_dot_dot, acc = qc1.inverse(yout[:, -1], new_distances, velocity,
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

def test_pothole_response():
    #sigma = 8  # 8 mm
    #profile_len, delta, cutoff_freq, delta2, seed = 100, .1, .15, .01, 55
    #dists, orig_hts, low_pass_hts, final_dists, final_heights = mp.make_gaussian(sigma, profile_len, delta,
    #                                                                            cutoff_freq, delta2, seed)

    final_dists, final_heights = mp.make_profile_from_psd('B', 'sine', .01, 100)
    #add pothole
    final_heights[2000:2000+20] = -40
    rp1 = rp.RoadProfile(final_dists, final_heights)

    velocity = 5  # 10 m/s
    m_s, m_u, c_s, k_s, k_u = 208, 28, 1300, 18709, 127200  # QC parameters
    qc1 = qc.QC(m_s, m_u, c_s, k_s, k_u)
    sample_rate_hz = 100
    T, yout, xout, new_distances, new_elevations = qc1.run(rp1, 100, velocity, sample_rate_hz)
    plt.plot(T, yout[:, -1])
    plt.title("Sprung mass acceleration from pothole")
    plt.show()
    #est_profile, est_xs, est_xs_dot, est_xu, est_xu_dot, est_xu_dot_dot, acc = qc1.inverse(yout[:, -1], new_distances, velocity,
    #                                                                                  sample_rate_hz)


    #iri_full = rp1.to_iri()
    #rp_est = rp.RoadProfile(new_distances, est_profile)
    #iri_orig = rp.RoadProfile(new_distances, new_elevations).to_iri()
    #iri_est = rp_est.to_iri()
    #plt.plot(new_distances, new_elevations, color='g')
    #plt.plot(new_distances, est_profile, color='r')
    #data1 = {"distances": new_distances, "elevations": new_elevations}
    #df1 = pd.DataFrame(data1)
    #data2 = {"distances": new_distances, "elevations": est_profile}
    #df2 = pd.DataFrame(data2)
    #df1.to_csv("/Users/gregoryislas/Documents/Mobilized/sampled_profile", index=False)
    #df2.to_csv("/Users/gregoryislas/Documents/Mobilized/estimated_profile", index=False)
    #plt.title("Estimated profile (r) (IRI={1}) and sampled input (IRI={2}, real={3}) (g), SR = {0}HZ".format(sample_rate_hz, np.round(iri_est, 3), np.round(iri_orig, 3), np.round(iri_full, 3)))
    #plt.show()

def show_response_diff_cars():
    #sigma = 8  # 8 mm
    #profile_len, delta, cutoff_freq, delta2, seed = 100, .1, .15, .01, 55
    #dists, orig_hts, low_pass_hts, final_dists, final_heights = mp.make_gaussian(sigma, profile_len, delta,
    #                                                                            cutoff_freq, delta2, seed)

    #final_dists, final_heights = mp.make_profile_from_psd('B', 'sine', .01, 100, seed=55)
    #add pothole
    car1 = qc.QC(m_s=240, m_u=36, k_s=16000, c_s=980, k_u=160000)
    car4 = qc.QC(m_s=300, m_u=50, k_s=18000, c_s=1200, k_u=180000)
    car5 = qc.QC(m_s=243, m_u=40, k_s=14671, c_s=370, k_u=124660)
    car6 = qc.QC(m_s=208, m_u=28, k_s=18709, c_s=1300, k_u=127200)
    bus = qc.QC(m_s=4500, m_u=500, k_s=300000, c_s=20000, k_u=1600000)
    suv1 = qc.QC(m_s=650, m_u=55, k_s=27500, c_s=3000, k_u=237000)
    suv2 = qc.QC(m_s=737.5, m_u=62.5, k_s=26750, c_s=3500, k_u=290000)
    golden = qc.QC(m_s=200, m_u=30, k_s=12660, c_s=1200, k_u=130600)
    golden2 = qc.QC(m_s=200*2, m_u=30*2, k_s=12660*2, c_s=1200*2, k_u=130600*2)
    golden3 = qc.QC(m_s=200 * .5, m_u=30 * .5, k_s=12660 * .5, c_s=1200 * .5, k_u=130600 * .5)

    list_vehicles = [car1, car4, car5, car6, bus, suv1, suv2]
    list_names = ["car1", "car4", "car5", "car6", "bus", "suv1", "suv2"]
    #rp1 = rp.RoadProfile(final_dists, final_heights)

    velocity = 5  # 10 m/s
    #m_s, m_u, c_s, k_s, k_u = 208, 28, 1300, 18709, 127200  # QC parameters
    #qc1 = qc.QC(m_s, m_u, c_s, k_s, k_u)
    sample_rate_hz = 500
    wavelen, amp, prof_len, delta = 20, 5, 100, .01
    for seed in range(0, 5):
        #final_dists, final_heights = mp.make_profile_from_psd('A', 'sine', .01, 100, seed=0)

        final_dists, final_heights = mp.make_sinusodal(wavelen, amp, prof_len, delta)
        rp1 = rp.RoadProfile(final_dists, final_heights)
        responses = []
        fig = plt.figure(figsize=(10, 10))
        ind = 1
        for v, n in zip(list_vehicles, list_names):

            T, yout, xout, new_distances, new_elevations = v.run(rp1, 100, velocity, sample_rate_hz)
            #responses.append((yout[:, -1] - np.mean(yout[:, -1]))/(np.std(yout[:, -1])))
            #plt.plot(T, (yout[:, -1])/(np.std(yout[:, -1])), label=n)
            #plt.plot(T, yout[:, -1], label=n)
            yout_true = qdeq.harmonic_solver(v.m_s, v.mu*v.m_s, v.c*v.m_s, v.k1*v.m_s, v.k2*v.m_s, amp/1000, wavelen, velocity, T)
            #plt.plot(T, (yout_true - np.mean(yout_true))/np.std(yout_true), label='{0} Analytic Solution'.format(n))
            #plt.plot(T, yout_true, label='{0} Analytic Solution'.format(n))
            ax = fig.add_subplot(3, 3, ind)
            #ax.hist(yout[:, -1], label='{0} Analytic Solution'.format(n))
            #plt.vlines(v.k2)
            yout_true = yout_true/v.k1**2
            ax.plot(T, yout_true, label=n)
            #ax.plot(T, yout[:, -1]/v.k1**2, label='Est')
            #ax.set_title("Min is {0}, max is {1}, \nvar is {2}".format(
            #    round(min(yout_true), 6), round(max(yout_true), 6), np.round(np.var(yout_true), 2)))
            ax.legend()
            #ax.set_xlim(-2, 2)
            #.show()
            #plt.title("A est vs. A true, v={0}".format(n))
            #responses.append(yout[:, -1])
            #plt.plot(T, yout[:, -1], label=n)
            ind += 1
        fig.tight_layout()
        plt.show()
        plt.plot(rp1.get_distances(), rp1.get_elevations())
        plt.title("Road profile, var = {0}".format(np.var(rp1.get_elevations()/1000)))
        plt.show()
        velocity += 5
        continue
        #T, yout, xout, new_distances, new_elevations = golden.run(rp1, 100, velocity, sample_rate_hz)
        yout_true = qdeq.harmonic_solver(golden.m_s, golden.mu * golden.m_s, golden.c * golden.m_s, golden.k1 * golden.m_s,
                                         golden.k2 * golden.m_s, amp / 1000,
                                         wavelen, velocity, T)

        plt.plot(T, yout_true, label='golden1')
        amp += 5
        plt.legend()
        plt.title("Analytic solution unstandardized for all cars sine")
        plt.show()
        #T, yout, xout, new_distances, new_elevations = golden.run(rp1, 100, velocity, sample_rate_hz)
        #plt.plot(T, (yout[:, -1] - np.mean(yout[:, -1]))/np.std(yout[:, -1]), label='golden')
        #plt.plot(T, yout[:, -1], label='golden1')
        #T, yout, xout, new_distances, new_elevations = golden2.run(rp1, 100, velocity, sample_rate_hz)
        # plt.plot(T, (yout[:, -1] - np.mean(yout[:, -1]))/np.std(yout[:, -1]), label='golden')
        #plt.plot(T, yout[:, -1], label='golden2')
        #T, yout, xout, new_distances, new_elevations = golden3.run(rp1, 100, velocity, sample_rate_hz)
        # plt.plot(T, (yout[:, -1] - np.mean(yout[:, -1]))/np.std(yout[:, -1]), label='golden')
        #plt.plot(T, yout[:, -1], label='golden3')
        #mn_responses = np.mean(np.array(responses), axis=0)


        #plt.title("Sprung mass acceleration from Diff Cars, seed = {0}".format(seed))
        #plt.legend()
        #plt.show()
        #plt.plot(T, mn_responses, label='mean')
        #plt.plot(T, (yout[:, -1] - np.mean(yout[: -1]))/np.std(yout[:, -1]), label='golden')

        #plt.legend()
        #plt.show()
        est_profile, x_s_dot, x_s, x_u, x_u_dot, x_u_dot_dot, est_distances, accelerations = golden.inverse(yout[:, -1],
                                                                                                 new_distances,
                                                                                                  velocity,
                                                                                                  sample_rate_hz, Wn=.0001)
       #print(est_distances)
        est_rp = rp.RoadProfile(est_distances, est_profile, filtered=True)
        iri_est = est_rp.to_iri()
        #plt.plot(est_rp.get_distances(), est_rp.get_elevations(), label='Est')
        #plt.plot(rp1.get_distances(), rp1.get_elevations(), label='True')
        #plt.title("Est Profile vs. True Profile,\n, True IRI={0}, Est={1}".format(rp1.to_iri(), iri_est))
        #plt.legend()
        #plt.show()
    #est_profile, est_xs, est_xs_dot, est_xu, est_xu_dot, est_xu_dot_dot, acc = qc1.inverse(yout[:, -1], new_distances, velocity,
    #                                                                                  sample_rate_hz)


    #iri_full = rp1.to_iri()
    #rp_est = rp.RoadProfile(new_distances, est_profile)
    #iri_orig = rp.RoadProfile(new_distances, new_elevations).to_iri()
    #iri_est = rp_est.to_iri()
    #plt.plot(new_distances, new_elevations, color='g')
    #plt.plot(new_distances, est_profile, color='r')
    #data1 = {"distances": new_distances, "elevations": new_elevations}
    #df1 = pd.DataFrame(data1)
    #data2 = {"distances": new_distances, "elevations": est_profile}
    #df2 = pd.DataFrame(data2)
    #df1.to_csv("/Users/gregoryislas/Documents/Mobilized/sampled_profile", index=False)
    #df2.to_csv("/Users/gregoryislas/Documents/Mobilized/estimated_profile", index=False)
    #plt.title("Estimated profile (r) (IRI={1}) and sampled input (IRI={2}, real={3}) (g), SR = {0}HZ".format(sample_rate_hz, np.round(iri_est, 3), np.round(iri_orig, 3), np.round(iri_full, 3)))
    #plt.show()

def print_iri_sinusoid():
    wavelen, amp, prof_len, delta = 20, 20, 100, .01
    for amp in range(5, 45, 5):
        final_dists, final_heights = mp.make_sinusodal(wavelen, amp, prof_len, delta)
        rp1 = rp.RoadProfile(final_dists, final_heights)
        print("IRI for amplitude {0} is {1}".format(amp, rp1.to_iri()))
#if __name__ == 'main':
print("NAME IS MAIN")
#inverse1()
#run_gaussian()
#test_inverse_gaussian()
show_response_diff_cars()
#print_iri_sinusoid()
