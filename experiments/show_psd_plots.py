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
    #new_freqs = orig_freqs * veloc * 2 * np.pi
    new_freqs = orig_freqs * veloc
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
    #new_freqs = orig_freqs * veloc * 2 * np.pi
    new_freqs = orig_freqs * veloc * 2* np.pi
    #t_func = new_freqs**2 * (cs*ku*new_freqs + ks*ku)/(mu*ms*new_freqs**4 + (cs*mu + cs*ms)*new_freqs**3 +
    #                                                   (ms*ks+mu*ks+ms*ku)*new_freqs**2 + (cs*ku)*new_freqs + ks*ku)
    t_func =(new_freqs**2*np.sqrt(cs**2*ku**2*new_freqs**2 + ks**2*ku**2))/np.sqrt(
        (cs*ku*new_freqs + new_freqs**3*(-cs*ms - cs*mu))**2 + (new_freqs**2*(-ks*ms-ks*mu - ku*ms)
                                                               + ks*ku + ms*mu*new_freqs**4)**2)

    return new_freqs/(2*np.pi), t_func**2*(orig_psd/veloc)

def get_transfer_func4(car, veloc, orig_freqs, orig_psd):
    ku = car.k1 * car.m_s
    ks = car.k2 * car.m_s
    ms = car.m_s
    mu = car.mu * car.m_s
    cs = car.c * car.m_s
    new_freqs = orig_freqs * veloc * 2 * np.pi
    #new_freqs = orig_freqs * veloc
    # t_func = new_freqs**2 * (cs*ku*new_freqs + ks*ku)/(mu*ms*new_freqs**4 + (cs*mu + cs*ms)*new_freqs**3 +
    #                                                   (ms*ks+mu*ks+ms*ku)*new_freqs**2 + (cs*ku)*new_freqs + ks*ku)
    t_func = (new_freqs ** 2 * np.sqrt(cs ** 2 * ku ** 2 * new_freqs ** 2 + ks ** 2 * ku ** 2)) / np.sqrt(
        (cs * ku * new_freqs + new_freqs ** 3 * (-cs * ms - cs * mu)) ** 2 + (
                    new_freqs ** 2 * (-ks * ms - ks * mu - ku * ms)
                    + ks * ku + ms * mu * new_freqs ** 4) ** 2)

    return t_func ** 2/veloc
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
    new_freqs = orig_freqs*veloc * 2* np.pi
    #psd2 = mp.iso_psd_function(gn0, new_freqs, .1*veloc)
    #new_freqs = orig_freqs * veloc
    denom = get_denom(new_freqs, car)
    num = get_num(new_freqs, car)
    new_psd = orig_psd/veloc * (new_freqs**4*num)/(denom**2) #Um - should be new_freqs**4 according to my calculations...
    return new_freqs/(2*np.pi), new_psd # I honestly don't know why the PSD is in angular frequency up to this point


def show_isoroughness_smooth():
    #sigma = 8 # should be a pretty smooth road
    #profile_len, delta, cutoff_freq, delta2, seed = 200, .3, .15, .1, 55

    #DEFINING THE PROFILE AND VEHICLE PARAMETERS
    profile_len = 100
    #dist, orig_hts, low_pass_hts, distances, elevations = mp.make_gaussian(sigma, profile_len, delta,
    #                                                                             cutoff_freq, delta2, seed)
    sample_rate_hz = 500
    dx = .05
    distances, elevations, true_gn0 = mp.make_profile_from_psd('B', 'sine', dx, profile_len, seed=2, ret_gn0=True)
    profile = rp.RoadProfile(distances, elevations, filtered=True)
    iri = profile.to_iri()
    vehicle = qc.QC(m_s=243, m_u=40, c_s=370, k_s=14671, k_u=124660)
    veloc = 10
    #T, yout, xout, new_dists, new_els = vehicle.run(profile, profile_len, veloc, sample_rate_hz)

    T, yout, xout, new_dists, new_els, vs = vehicle.run2(profile, [], veloc, final_sample_rate=sample_rate_hz)
    print(T)


    #freqs, psd = cisr.compute_psd(profile, dx=np.diff(profile.get_distances())[0])

    #PLOTTING THE ORIGINAL PROFILE
    plt.plot(distances, elevations)
    #plt.ylim(-50, 50)
    plt.title("Original Road Profile")
    plt.show()






    #PLOTTING THE POWER SPECTRAL DENSITY OF THE ROAD PROFILE, WHAT SHOULD BE VS. CALCULATED ONE FROM PROFILE ITSELF
    freqs = np.arange(.001, 20, .05)
    psd = mp.iso_psd_function(true_gn0, freqs)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 2, 1)
    ax.loglog(freqs, psd, label='Hypothetical PSD')
    #print(elevations)
    #f_road, psd_road = welch(elevations/1000, 1/dx, scaling='density', nperseg=4096)
    f_road, psd_road = periodogram(elevations/1000, 1/dx, scaling='density')
    #f_road2, psd_road2 = periodogram(elevations / 1000, 1/dx, scaling='spectrum')
    #print(f_road)
    #print(freqs)
    #print(psd)
    #print(psd_road)
    ax.set_ylim(1e-10, 1)
    ax.legend()
    ax = fig.add_subplot(2, 2, 2)
    ax.loglog(f_road, psd_road, label='Calculated PSD density')
    ax.loglog(freqs, psd, label='Hypothetical PSD')
    ax.set_ylim(1e-10, 1)
    ax.legend()

    psd_w = mp.iso_psd_function_angular(true_gn0/16, freqs/(2*np.pi))
    ax = fig.add_subplot(2, 2, 3)
    ax.loglog(f_road, psd_road, label='Calculated PSD density')
    ax.loglog(freqs/(2*np.pi), psd_w, label='Hypothetical PSD')
    ax.legend()
    ax.set_ylim(1e-10, 1)

    #plt.legend()
    #plt.title("PSD of Original Road Profile")
    #plt.xlim(1e-2, 10)
    # plt.ylim(1e-10, 1)
    fig.tight_layout()
    plt.show()

    f_time, psd_time = periodogram(elevations / 1000, 1 / dx * veloc)
    plt.loglog(f_time, psd_time, label='True PSD Time Domain')
    f_time_qc, psd_time_qc = periodogram(new_els / 1000, sample_rate_hz)
    plt.loglog(f_time_qc, psd_time_qc, label='QC PSD Time Domain')
    plt.loglog(f_road*veloc, psd_road/veloc, label='Estimated PSD Time Domain')
    plt.legend()
    plt.title("PSD in time domain of road profile")

    plt.show()


    #CREATING THE SMOOTHED POWER SPECTRAL DENSITY
    smth_freqs, smth_psd = cisr.smooth_psd(f_road, psd_road)
    regressor = cisr.fit_smooth_psd(smth_freqs, smth_psd)

    # print("lr coefficient is {0}, lr intercept is {1}".format(regressor.coef_, regressor.intercept_))
    road_class = regressor.coef_[0][0]* 1e6/.01
    #preds = regressor.predict(np.arange(0, 10, .001).reshape(-1, 1))
    print(
        "regressor coef {0}, regressor intercept {1},  est gn0 is {2}, true gn0 {3}, iri is {4}".format(regressor.coef_,
                                                                                                        regressor.intercept_,
                                                                                                        road_class,
                                                                                                        true_gn0, iri))

    #plt.loglog(f_road2, psd_road2, label='Calculated PSD spectrum')
    #plt.plot(freqs, psd)

    #PLOTTING THE SMOOTHED PSD VS. TRUE PSD AND REGRESSED PSD AND MAX PSD FOR ROAD CLASS
    plt.loglog(freqs, psd, label='Hypothetical PSD')
    plt.loglog(smth_freqs, smth_psd, label='Smoothed PSD')

    #to_predict = (smth_freqs)**(-2)
    #to_plot = np.arange(0, 10, .1)
    #print("to predict is {0}".format(to_predict))
    preds1 = regressor.coef_[0][0] * (np.arange(.001, 10, .01))**-2#to_predict#regressor.predict(to_predict.reshape(-1, 1))
    preds1 = preds1.flatten()

    plt.loglog(np.arange(0, 10, .01), preds1, label='Regressed PSD')


    #plt.plot(smth_freqs, regressor.predict(smth_freqs.reshape(-1, 1)))
    g_n0, n0 = 128, .1
    preds2 = g_n0 * 1e-6 * (smth_freqs / n0) ** (-2)
    #plt.loglog(smth_freqs, preds2, color='r', label='Max PSD for class')
    #plt.loglog(f_road, psd_road, label='Orig PSD')
    plt.title("Smoothed PSD of Original Road Profile")
    plt.legend()
    #plt.xlim(1e-2, 10)
    plt.ylim(1e-10, 10)
    plt.show()


    #PLOTTING THE PSD OF THE ACCELERATION SERIES
    #f, acc_psd = welch(yout[:, -1], sample_rate_hz, scaling='density', nperseg=int(2*1/(veloc*.01)*sample_rate_hz))
    f, acc_psd = periodogram(yout[:, -1], sample_rate_hz, scaling='density')

    plt.loglog(f, acc_psd)
    plt.title("Acc PSD")
    plt.ylim(1e-10, 10)
    plt.show()

    # PLOTTING THE acceleration series OF THE ACCELERATION SERIES
    #f, acc_psd = periodogram(yout[:, -1], sample_rate_hz, scaling='density')
    #plt.loglog(f, acc_psd)
    #plt.title("Acc PSD")
    #plt.ylim(1e-10, 10)
    plt.plot(T, yout[:, -1])
    plt.title("Raw acceleration series")
    plt.show()


    #PLOTTING THE TWO TRANSFER FUNCTIONS FOR COMPARISON
    new_f, new_psd = acc_transfer_function(vehicle, veloc, f_road, psd_road)
    #plt.loglog(new_f (2 * np.pi), new_psd,  label='My Transfer')
    plt.loglog(new_f, new_psd, label='My Transfer')
    plt.ylim(1e-10, 10)
    #plt.title("Transfer PSD")
    #plt.show()
    newt_f, newt_psd = get_transfer_func3(vehicle, veloc, f_road, psd_road)
    #plt.loglog(newt_f / (2 * np.pi), newt_psd, label='Transfer 3')
    plt.loglog(newt_f, newt_psd, label='Transfer 3')
    #plt.ylim(1e-10, 10)
    #plt.title("Transfer PSD 3")
    plt.title("My Transfer vs Transfer 3")
    plt.legend()
    plt.show()


    #PLOTTING FIRST TRANSFER FUNCTION VS. COMPUTED ACC PSD
    plt.loglog(f, acc_psd, label='Acc PSD')
    #plt.loglog(new_f / (2 * np.pi), new_psd,  'o', label='Transfer PSD')
    plt.loglog(new_f, new_psd, label='Transfer PSD')
    plt.legend()
    plt.title("Acc PSD vs Orig Transfer func")
    #plt.ylim(1e-10, 10)
    plt.show()
    d1 = acc_psd[:min(len(new_psd), len(acc_psd))]
    d2 = new_psd[:min(len(new_psd), len(acc_psd))]
    rat = np.median(d2[np.where(d1 != 0)]/d1[np.where(d1 != 0)])
    indices = np.where(d2[np.where(d1 != 0)]/d1[np.where(d1 != 0)] >= 1000)
    print(f[indices])
    print(acc_psd[indices])
    print("")
    print(new_f[indices])
    print(new_psd[indices])
    print()
    print("")
    print(f[indices[0] + 1])
    print(acc_psd[indices[0] + 1])
    print("")
    plt.plot(d2[np.where(d1 != 0)]/d1[np.where(d1 != 0)])
    plt.ylim(0, 300)
    plt.title("Plot of ratio, veloc = {0}".format(veloc))
    plt.show()
    print("Ratio transfer function to acc psd: {0}".format(
        round(rat, 3)))
    print("freq resolution acc: {0}, freq resolution transfer: {1}".format(f[1] - f[0], new_f[1] - new_f[0]))
    print("len acc_psd: {0}, len new_psd: {1}".format(len(acc_psd), len(new_psd)))
    print(f)
    print(new_f)


    #rat = 0
    #rat = np.mean(acc_psd[np.where(new_psd != 0)]/new_psd[np.where(new_psd != 0)])
    #print("Len acc_psd, len new_psd are {0}, {1}".format(len(acc_psd), len(new_psd)))

    #PLOTTING SECOND TRANSFER FUNCTION VS. COMPUTED ACC PSD
    #plt.loglog(newt_f / (2 * np.pi), newt_psd, label='Transfer PSD3')
    plt.loglog(f, acc_psd, label='Acc PSD')
    plt.loglog(newt_f, newt_psd, label='Transfer PSD3')


    rat = 0
    # rat = np.mean(acc_psd[np.where(new_psd != 0)]/new_psd[np.where(new_psd != 0)])
    # print("Len acc_psd, len new_psd are {0}, {1}".format(len(acc_psd), len(new_psd)))
    plt.title("Acceleration PSD vs Transfer PSD 3")
    plt.legend()
    plt.ylim(1e-10, 10)
    plt.show()

    #PLOTTING RATIO OF ACCPSD TO ROAD PSD VS. ESTIMATED RATIO BASED ON TRANSFER FUNCTION
    plt.loglog(f_road[:min(len(psd_road), len(acc_psd))],
               acc_psd[:min(len(psd_road), len(acc_psd))]/psd_road[:min(len(psd_road), len(acc_psd))], label="True ratio")
    #plt.loglog(new_f/(2* np.pi), (f_road*veloc*2*np.pi)**4*get_num(car=vehicle, freqs=f_road*veloc*2*np.pi)/(get_denom(car=vehicle, freqs=f_road*veloc*2*np.pi)**2),
    #           label="Est Ratio")
    plt.loglog(new_f,
               ((f_road * veloc * 2*np.pi) ** 4 * get_num(car=vehicle, freqs=f_road) / (
                           get_denom(car=vehicle, freqs=f_road) ** 2))/(2*np.pi)**2,
               label="Est Ratio")
    plt.title("What transfer func should be...")
    plt.legend()
    plt.show()


    #PLOTTING THE ROAD PSD VS. THE PSD OF THE ROAD BASED ON DIVIDING THE ACC PSD BY THE TRANSFER FUNCTIONS
    plt.loglog(f_road, psd_road, label='Road PSD')
    plt.loglog(f/veloc,
               acc_psd / (((f*2*np.pi)**4*get_num(car=vehicle, freqs=f*2*np.pi)/(get_denom(car=vehicle, freqs=f*2*np.pi)**2))/(2*np.pi)**2),
               label="Recovered PSD")
    #plt.loglog(f / (veloc),
    #           acc_psd / acc_transfer_function(vehicle, veloc, f, psd_road)[1],
    #           label="Recovered PSD")
    #plt.loglog(f / veloc,
    #           acc_psd / get_transfer_func4(vehicle, veloc, f / veloc, None),
    #           label="Recovered PSD Transfer 2")
    #plt.loglog(f / (veloc),
    #           acc_psd / get_transfer_func4(vehicle, veloc, f, None),
    #           label="Recovered PSD Transfer 2")
    # plt.plot(freqs, psd)
    plt.legend()
    plt.title("PSD of Original Road Profile vs Recovered(s)")
    plt.show()


    print("regressor coef {0}, regressor intercept {1}".format(regressor.coef_, regressor.intercept_))








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


def show_plot_with_constantacc():
    dx = .05
    distances, elevations, true_gn0 = mp.make_profile_from_psd('B', 'sine', .05, 200, seed=69, ret_gn0=True)
    profile = rp.RoadProfile(distances, elevations)
    plt.plot(distances, elevations)
    # plt.ylim(-50, 50)
    plt.title("Original Road Profile")
    plt.show()

    v0 = 0
    vehicle = qc.QC(m_s=243, m_u=40, c_s=370, k_s=14671, k_u=124660)
    #freqs, psd = cisr.compute_psd(profile, dx=np.diff(profile.get_distances())[0])
    freqs = np.arange(.001, 20, .05)
    psd = mp.iso_psd_function(true_gn0, freqs)
    plt.loglog(freqs, psd, label='Hypothetical PSD')
    # print(elevations)
    f_road, psd_road = periodogram(elevations / 1000, 1/dx)
    # print(f_road)
    # print(freqs)
    # print(psd)
    # print(psd_road)
    plt.loglog(f_road, psd_road, label='Calculated PSD')
    # plt.plot(freqs, psd)
    plt.legend()
    plt.title("PSD of Original Road Profile")
    #plt.xlim(1e-2, 10)
    plt.ylim(1e-10, 1)
    plt.show()

    sample_rate_hz = 1000
    veloc = 10
    T1, yout1, xout1, new_dists1, new_els1 = vehicle.run(profile, profile.length(), veloc, sample_rate_hz)

    T, yout, xout, new_dists, new_els, vs = vehicle.run2(profile, [(2, 5), (5 / 2, 2), (-5/2, 2)], v0=v0, final_sample_rate=sample_rate_hz)

    f, acc_psd = periodogram(yout[:, -1], sample_rate_hz)
    f1, acc_psd1 = periodogram(yout1[:, -1], sample_rate_hz)
    plt.loglog(f, acc_psd, label='PSD W/Varying Veloc')
    plt.loglog(f1, acc_psd1, label='PSD Constant Veloc')
    plt.title("Acc PSD")
    plt.legend()
    plt.ylim(1e-10, 10)
    plt.show()

def show_psd_time_domain():
    profile_len = 500
    # dist, orig_hts, low_pass_hts, distances, elevations = mp.make_gaussian(sigma, profile_len, delta,
    #                                                                             cutoff_freq, delta2, seed)
    #sample_rate_hz = 250
    dx = .05
    distances, elevations, true_gn0 = mp.make_profile_from_psd('B', 'sine', dx, profile_len, seed=69, ret_gn0=True)
    profile = rp.RoadProfile(distances, elevations, filtered=True)
    freqs = np.arange(.001, 20, .05)
    psd = mp.iso_psd_function(true_gn0, freqs)
    plt.plot(distances, elevations)
    # plt.ylim(-50, 50)
    plt.title("Original Road Profile")
    plt.show()
    f_road, psd_road = periodogram(elevations / 1000, 1 / dx)
    plt.loglog(freqs, psd, label='Hypothetical PSD')
    plt.loglog(f_road, psd_road, label='Calculated PSD')
    # plt.plot(freqs, psd)
    plt.legend()
    plt.title("PSD of Original Road Profile")
    # plt.xlim(1e-2, 10)
    plt.ylim(1e-10, 1)
    plt.show()
    veloc = 25
    times = distances/veloc
    plt.plot(times, elevations)
    plt.title("Elevations in time domain")
    plt.show()
    f_time, psd_time = periodogram(elevations/1000, 1/dx * veloc)
    plt.loglog(f_time, psd_time)
    plt.title("PSD in time domain of road profile")
    plt.show()

    new_f = f_road*veloc
    plt.loglog(f_time, psd_time, label='Computed PSD in time domain')
    plt.loglog(new_f, psd_road/veloc, label='Estimated PSD in time domain')
    plt.loglog(f_road, psd_road, label='Road PSD')
    plt.legend()
    plt.show()
    d1 = psd_time[:min(len(psd_time), len(psd_road))]
    d2 = psd_road[:min(len(psd_time), len(psd_road))]
    rat = np.median(d2[np.where(d1 != 0)]/d1[np.where(d1 != 0)])
    print("rat is {0}".format(round(rat, 4)))

def show_car_transfer_function():
    profile_len = 500
    # dist, orig_hts, low_pass_hts, distances, elevations = mp.make_gaussian(sigma, profile_len, delta,
    #                                                                             cutoff_freq, delta2, seed)
    sample_rate_hz = 250
    dx = .05
    distances, elevations, true_gn0 = mp.make_profile_from_psd('B', 'sine', dx, profile_len, seed=69, ret_gn0=True)
    profile = rp.RoadProfile(distances, elevations, filtered=True)
    freqs = np.arange(.001, 20, .05)
    psd = mp.iso_psd_function(true_gn0, freqs)
    plt.plot(distances, elevations)
    # plt.ylim(-50, 50)
    plt.title("Original Road Profile")
    plt.show()
    f_road, psd_road = periodogram(elevations / 1000, 1 / dx)
    plt.loglog(freqs, psd, label='Hypothetical PSD')
    plt.loglog(f_road, psd_road, label='Calculated PSD')
    # plt.plot(freqs, psd)
    plt.legend()
    plt.title("PSD of Original Road Profile")
    # plt.xlim(1e-2, 10)
    plt.ylim(1e-10, 1)
    plt.show()

    vehicle = qc.QC(m_s=243, m_u=40, c_s=370, k_s=14671, k_u=124660)
    veloc = 15
    T, yout, xout, new_dists, new_els, vs = vehicle.run2(profile, [], veloc, final_sample_rate=sample_rate_hz)

    f_rawacc, psd_rawacc = periodogram(yout[:, -1], sample_rate_hz)

    f_trans, psd_trans = vehicle.transfer_function(f_road, psd_road, veloc)

    plt.loglog(f_rawacc, psd_rawacc, label='Raw Acceleration PSD')
    plt.loglog(f_trans, psd_trans, label='Transfer Function PSD')
    plt.legend()
    plt.title("Transfer function PSD vs. Acc Time Series PSD")
    plt.show()

    f_road2, psd_road2 = vehicle.inverse_transfer_function(f_rawacc, psd_rawacc, veloc)
    plt.loglog(f_road, psd_road, label='Road PSD')
    plt.loglog(f_road2, psd_road2, label='PSD From Transfer Function')
    plt.legend()
    plt.title('Road PSD vs. Recovered PSD')
    plt.show()



#show_plot_with_constantacc()
show_isoroughness_smooth()
#show_car_acc_plots()
#show_psd_time_domain()
#show_car_transfer_function()