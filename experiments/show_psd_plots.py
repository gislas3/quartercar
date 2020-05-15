from sklearn.linear_model import LinearRegression
import numpy as np
from tests import make_profile as mp
from experiments import computeisoroughness as cisr
from quartercar import roadprofile as rp
from matplotlib import pyplot as plt
from quartercar import qc
import pandas as pd
from scipy.signal import periodogram, welch



def get_transfer_func2(car, veloc, orig_freqs, orig_psd):
    ku = car.k1 * car.m_s
    ks = car.k2 * car.m_s
    ms = car.m_s
    mu = car.mu * car.m_s
    cs = car.c * car.m_s
    new_freqs = orig_freqs * veloc * 2 * np.pi
    num1 = ms*new_freqs**2*ku
    denom1 = ku - ((ks + new_freqs*cs)*new_freqs**2*ms/(-ms*new_freqs**2) + ks + new_freqs*cs) - mu*new_freqs**2
    num2 = ms*new_freqs**2
    denom2 = ks - ms*new_freqs**2+ new_freqs*cs
    t_func = num1/denom1 * (1 + num2/denom2)
    return new_freqs, orig_psd*(new_freqs**4)*t_func/(ms**2)

def get_transfer_func3(car, veloc, orig_freqs, orig_psd):
    ku = car.k1 * car.m_s
    ks = car.k2 * car.m_s
    ms = car.m_s
    mu = car.mu * car.m_s
    cs = car.c * car.m_s
    new_freqs = orig_freqs * veloc * 2 * np.pi
    #t_func = new_freqs**2 * (cs*ku*new_freqs + ks*ku)/(mu*ms*new_freqs**4 + (cs*mu + cs*ms)*new_freqs**3 +
    #                                                   (ms*ks+mu*ks+ms*ku)*new_freqs**2 + (cs*ku)*new_freqs + ks*ku)
    t_func = np.sqrt(cs**2*ku**2*new_freqs**6 + ks**2*mu**2*new_freqs**4)/np.sqrt(
        (cs*ks*new_freqs + new_freqs**3*(-cs*ms - cs*mu))**2 + (new_freqs**2*(-ks*ms-ks*mu - ku*ms)
                                                               + ks*ku + ms*mu*new_freqs**4)**2)

    return new_freqs, t_func**2*orig_psd

def get_transfer_func4(car, veloc, orig_freqs, orig_psd):
    ku = car.k1 * car.m_s
    ks = car.k2 * car.m_s
    ms = car.m_s
    mu = car.mu * car.m_s
    cs = car.c * car.m_s
    new_freqs = orig_freqs * veloc * 2 * np.pi
    # t_func = new_freqs**2 * (cs*ku*new_freqs + ks*ku)/(mu*ms*new_freqs**4 + (cs*mu + cs*ms)*new_freqs**3 +
    #                                                   (ms*ks+mu*ks+ms*ku)*new_freqs**2 + (cs*ku)*new_freqs + ks*ku)
    t_func = np.sqrt(cs ** 2 * ku ** 2 * new_freqs ** 6 + ks ** 2 * mu ** 2 * new_freqs ** 4) / np.sqrt(
            (cs * ks * new_freqs + new_freqs ** 3 * (-cs * ms - cs * mu)) ** 2 + (
                        new_freqs ** 2 * (-ks * ms - ks * mu - ku * ms)
                        + ks * ku + ms * mu * new_freqs ** 4) ** 2)

    return t_func ** 2
           #*(1 + ms*new_freqs**2/(ks - ms*new_freqs**2 + new_freqs*cs))

def get_denom(freqs, car):
    ku = car.k1 * car.m_s
    ks = car.k2 * car.m_s
    ms = car.m_s
    mu = car.mu * car.m_s
    cs = car.c * car.m_s
    #freqs = freqs*2*np.pi
    denom = ku**2*(cs**2*freqs**2 + ks**2 - 2*ks*ms*freqs**2 + ms**2 * freqs**4) - \
            2*ku*(cs**2 * freqs**4*(ms + mu) + ks**2*freqs**2*(ms + mu) - ks*ms*freqs**4*(ms + 2*mu)
                  + ms**2*mu*freqs**6) + freqs**6*(cs**2 * (ms + mu)**2 + ms**2*mu**2*freqs**2) + \
                ks**2*freqs**4*(ms + mu)**2 - 2*ks*ms*mu*freqs**6*(ms + mu)
    return denom




def get_num(freqs, car):
    ku = car.k1 * car.m_s
    ks = car.k2 * car.m_s
    ms = car.m_s
    mu = car.mu * car.m_s
    cs = car.c * car.m_s
    #freqs = freqs*2*np.pi*
    alpha = (-(cs**2)*ms*freqs**4 - cs**2*mu*freqs**4 + ks*ms*mu*freqs**4 + cs**2*ku*freqs**2 -
            ks**2*ms*freqs**2 - ku*ks*ms*freqs**2 - ks**2*mu*freqs**2 + ku*ks**2)*ku
    beta = (cs*ku*ms*freqs**3 - cs*ms*mu*freqs**5)*ku
    return alpha**2 + beta**2

def acc_transfer_function(car, veloc, orig_freqs, orig_psd):
    new_freqs = orig_freqs*veloc*2*np.pi
    denom = get_denom(new_freqs, car)
    num = get_num(new_freqs, car)
    new_psd = orig_psd * (new_freqs**4*num)/(denom**2) #Um - should be new_freqs**4 according to my calculations...
    return new_freqs, new_psd


def show_isoroughness_smooth():
    sigma = 8 # should be a pretty smooth road
    profile_len, delta, cutoff_freq, delta2, seed = 200, .3, .15, .1, 55
    dist, orig_hts, low_pass_hts, distances, elevations = mp.make_gaussian(sigma, profile_len, delta,
                                                                                 cutoff_freq, delta2, seed)
    sample_rate_hz = 100
    #distances, elevations = mp.make_profile_from_psd('C', 'sine', .1, 500, seed=55)
    profile = rp.RoadProfile(distances, elevations, filtered=True)
    iri = profile.to_iri()
    vehicle = qc.QC(m_s=243, m_u=40, c_s=370, k_s=14671, k_u=124660)
    veloc = 20
    T, yout, xout, new_dists, new_els = vehicle.run(profile, 1000, veloc, sample_rate_hz)
    freqs, psd = cisr.compute_psd(profile, dx=np.diff(profile.get_distances())[0])
    smth_freqs, smth_psd = cisr.smooth_psd(freqs, psd)
    regressor = cisr.fit_smooth_psd(smth_freqs, smth_psd)
    #print("lr coefficient is {0}, lr intercept is {1}".format(regressor.coef_, regressor.intercept_))
    road_class = regressor.coef_[0][0]/.01
    plt.plot(distances, elevations)
    #plt.ylim(-50, 50)
    plt.title("Original Road Profile")
    plt.show()

    plt.loglog(freqs, psd, label='Hypothetical PSD')
    #print(elevations)
    f_road, psd_road = periodogram(elevations/1000, 10)
    #print(f_road)
    #print(freqs)
    #print(psd)
    #print(psd_road)
    plt.loglog(f_road, psd_road, label='Calculated PSD')
    #plt.plot(freqs, psd)
    plt.legend()
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
    f, acc_psd = periodogram(yout[:, -1], sample_rate_hz)
    plt.loglog(f, acc_psd)
    plt.title("Acc PSD")
    plt.ylim(1e-10, 10)
    plt.show()
    new_f, new_psd = acc_transfer_function(vehicle, veloc, f_road, psd_road)
    plt.loglog(new_f / (2 * np.pi), new_psd,  label='My Transfer')
    plt.ylim(1e-10, 10)
    #plt.title("Transfer PSD")
    #plt.show()
    newt_f, newt_psd = get_transfer_func3(vehicle, veloc, f_road, psd_road)
    plt.loglog(newt_f / (2 * np.pi), newt_psd, label='Transfer 3')
    #plt.ylim(1e-10, 10)
    #plt.title("Transfer PSD 3")
    plt.title("My Transfer vs Transfer 3")
    plt.legend()
    plt.show()

    plt.loglog(f, acc_psd, 'o', label='Acc PSD')
    plt.loglog(new_f / (2 * np.pi), new_psd,  'o', label='Transfer PSD')
    plt.legend()
    plt.title("Acc PSD vs Orig Transfer func")
    plt.ylim(1e-10, 10)
    plt.show()
    rat = 0
    #rat = np.mean(acc_psd[np.where(new_psd != 0)]/new_psd[np.where(new_psd != 0)])
    #print("Len acc_psd, len new_psd are {0}, {1}".format(len(acc_psd), len(new_psd)))


    plt.loglog(newt_f / (2 * np.pi), newt_psd, label='Transfer PSD3')

    plt.loglog(f, acc_psd, label='Acc PSD')

    rat = 0
    # rat = np.mean(acc_psd[np.where(new_psd != 0)]/new_psd[np.where(new_psd != 0)])
    # print("Len acc_psd, len new_psd are {0}, {1}".format(len(acc_psd), len(new_psd)))
    plt.title("Acceleration PSD vs Transfer PSD 3")
    plt.legend()
    plt.ylim(1e-10, 10)
    plt.show()

    plt.loglog(f_road[:min(len(psd_road), len(acc_psd))] * veloc,
               acc_psd[:min(len(psd_road), len(acc_psd))]/psd_road[:min(len(psd_road), len(acc_psd))], label="True ratio")
    plt.loglog(new_f/(2* np.pi), (f_road*veloc*2*np.pi)**4*get_num(car=vehicle, freqs=f_road*veloc*2*np.pi)/(get_denom(car=vehicle, freqs=f_road*veloc*2*np.pi)**2),
               label="Est Ratio")
    plt.title("What transfer func should be...")
    plt.legend()
    plt.show()

    #Last plot... show road divided by the transfer function
    plt.loglog(f_road, psd_road, label='Road PSD')
    plt.loglog(f/veloc,
               acc_psd / ((f*2*np.pi)**4*get_num(car=vehicle, freqs=f*2*np.pi)/(get_denom(car=vehicle, freqs=f*2*np.pi)**2)),
               label="Recovered PSD")
    plt.loglog(f / veloc,
               acc_psd / get_transfer_func4(vehicle, veloc, f / veloc, None),
               label="Recovered PSD Transfer 2")
    # plt.plot(freqs, psd)
    plt.legend()
    plt.title("PSD of Original Road Profile vs Recovered(s)")
    plt.show()

    #plt.loglog(f,new_psd[:len(acc_psd)] - acc_psd)
    #plt.plot(f,new_psd[:len(acc_psd)] - acc_psd)
    #plt.ylim(0, 1)
    #plt.title("Diff of transfer pds vs computed psd")
    #plt.show()

    #plt.title("Transfer Acceleration PSD")
    #plt.ylim(1e-10, 10)
    #plt.show()
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

show_isoroughness_smooth()
#show_car_acc_plots()