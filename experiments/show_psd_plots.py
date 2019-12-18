from sklearn.linear_model import LinearRegression
import numpy as np
from tests import make_profile as mp
from experiments import computeisoroughness as cisr
from quartercar import roadprofile as rp
from matplotlib import pyplot as plt
from quartercar import qc
import pandas as pd

def show_isoroughness_smooth():
    sigma = 8 # should be a pretty smooth road
    #profile_len, delta, cutoff_freq, delta2, seed = 1000, .3, .15, .01, 55
    #dists, orig_hts, low_pass_hts, final_dists, final_heights = mp.make_gaussian(sigma, profile_len, delta,
     #                                                                            cutoff_freq, delta2, seed)
    distances, elevations = mp.make_profile_from_psd('B', 'sine', .1, 100)
    profile = rp.RoadProfile(distances, elevations)
    iri = profile.to_iri()
    freqs, psd = cisr.compute_psd(profile, dx=np.diff(profile.get_distances())[0])
    smth_freqs, smth_psd = cisr.smooth_psd(freqs, psd)
    regressor = cisr.fit_smooth_psd(smth_freqs, smth_psd)
    #print("lr coefficient is {0}, lr intercept is {1}".format(regressor.coef_, regressor.intercept_))
    road_class = regressor.coef_[0][0]/.01
    plt.plot(distances, elevations)
    #plt.ylim(-50, 50)
    plt.title("Original Road Profile")
    plt.show()

    plt.loglog(freqs, psd)
    #plt.plot(freqs, psd)
    plt.title("PSD of Original Road Profile")
    plt.xlim(1e-2, 10)
    plt.ylim(1e-10, 1)
    plt.show()

    plt.loglog(smth_freqs, smth_psd)
    to_predict = (smth_freqs)**(-2)
    #to_plot = np.arange(0, 10, .1)
    #print("to predict is {0}".format(to_predict))
    preds1 = regressor.coef_[0][0] * to_predict#regressor.predict(to_predict.reshape(-1, 1))
    preds1 = preds1.flatten()

    plt.loglog(smth_freqs, preds1)

    #plt.plot(smth_freqs, regressor.predict(smth_freqs.reshape(-1, 1)))
    g_n0, n0 = 32, .1
    preds2 = g_n0 * 1e-6 * (smth_freqs / n0) ** (-2)
    plt.loglog(smth_freqs, preds2, color='r')
    plt.title("Smoothed PSD of Original Road Profile")
    #plt.xlim(1e-2, 10)
    plt.ylim(1e-10, 10)
    plt.show()
    #print("preds1 is {0}".format(preds1))
    #print("preds2 is {0}".format(preds2))
    print("regressor coef {0}, regressor intercept {1}".format(regressor.coef_, regressor.intercept_))



    #x = np.arange(0, 10, .1)
    #y = 5*x**2
    #plt.plot(x, y)
    #lr = LinearRegression()
    #lr.fit((x**2).reshape(-1, 1), y)
    #plt.plot(x, lr.predict((x**2).reshape(-1, 1)))
    #plt.show()
    #print("lr coef is {0}".format(lr.coef_))
    print("Road class is {0}, IRI is {1}".format(road_class/1e-6, iri))

def show_car_acc_plots():
    distances, elevations = mp.make_profile_from_psd('C', 'sine', .05, 100)
    profile = rp.RoadProfile(distances, elevations)
    iri = profile.to_iri()
    vehicle = qc.QC(m_s=240, m_u=36, c_s=980, k_s=16000, k_u=160000)
    T, yout, xout, new_dists, new_els = vehicle.run(profile, 100, 10, 100)
    accs = yout[:, -1]
    plt.plot(new_dists, accs)
    plt.title("Plot of distances vs accelerations of sprung mass, velocity 10 m/s")
    plt.show()
    plt.plot(new_dists, new_els)
    plt.title("Original profile, iri is {0}".format(iri))
    plt.show()
    df = pd.DataFrame({'distances': new_dists, 'accelerations': accs})
    df.to_csv("/Users/gregoryislas/Documents/Mobilized/data_for_iri_experimenter.csv", index=False)

#show_isoroughness_smooth()
show_car_acc_plots()