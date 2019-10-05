from quartercar import qc

import tests.make_profile as mp
#from matplotlib import pyplot as plt
from quartercar import roadprofile as rp
#from scipy.fftpack import fft
#from scipy import signal
import numpy as np
import pandas as pd
from scipy import signal, integrate


#Step 1: Generate 100 road profiles of 100 meters each
sigma_seed = 15
np.random.seed(sigma_seed)

#smooth road profiles:
sigmas1 = np.abs(np.random.normal(6, 3, 33))

#rougher road profiles
sigmas2 = np.abs(np.random.normal(15, 2, 33))

#even rougher road profiles
sigmas3 = np.abs(np.random.normal(25, 2, 34))


road_profiles = {}

profile_len, delta, cutoff_freq, delta2 = 100, .1, .15, .01

for x in range(0, 100):
    seed = x
    if(x <= 32):
        sigma = sigmas1[x]
    elif(x > 32 and x <= 65):
        sigma = sigmas2[x-33]
    else:
        sigma = sigmas3[x-66]
    dists, orig_hts, low_pass_hts, final_dists, final_heights = mp.make_gaussian(sigma, profile_len, delta,
                                                                                 cutoff_freq, delta2, x)
    profile = rp.RoadProfile(final_dists, final_heights)
    #this is to double check IRIs are properly calculated
    #df1 = pd.DataFrame({"dists": profile.get_distances(), "elevations": profile.get_elevations()})
    #df1.to_csv("/Users/gregoryislas/Documents/Mobilized/Test_Profiles/{0}.csv".format(x), index=False)
    #df1 = None

    road_profiles[x] = profile

#iris1, iris2, iris3 = [], [], []
#here we go...
#use usual parameters for QC model:

m_s, m_u, c_s, k_s, k_u = 208, 28, 1300, 18709, 127200  # QC parameters
qc1 = qc.QC(m_s, m_u, c_s, k_s, k_u)
#What we shall test:
#velocity 1 - 40 m/s
#sample rate: 50, 100, 200, 500 HZ

for velocity in range(5, 45, 5):
    for sample_rate in [50, 100, 200, 500]:
        output_str = "File_Name,True_IRI,Pred_IRI,Pred_IRI_FILT\n"
        for x in range(0, 100):
            profile = road_profiles[x]
            true_iri = profile.to_iri()
            T, yout, xout, new_distances, new_elevations = qc1.run(profile, 100, velocity, sample_rate)
            accelerations = yout[:, -1]
            velocities = integrate.cumtrapz(accelerations, T, initial=0)
            displacements = integrate.cumtrapz(velocities, T, initial=0)
            displacements = displacements*1000
            road_profile_proxy = rp.RoadProfile(new_distances, displacements)
            pred_iri = road_profile_proxy.to_iri()

            sample_rate_per_meter = int(sample_rate / velocity)
            longest_wave_len = 91  # longest wavelength of interest

            sos = signal.butter(10, 1 / longest_wave_len, 'highpass', fs=sample_rate_per_meter, output='sos')
            road_profile_proxy_filt = rp.RoadProfile(new_distances, signal.sosfilt(sos, displacements))
            pred_iri_filt = road_profile_proxy.to_iri()
            output_str += "{0},{1},{2},{3}\n".format(x, true_iri, pred_iri, pred_iri_filt)
        with open("/Users/gregoryislas/Documents/Mobilized/IRI_Exp_Files/Vel_{0}_Sr_{1}.csv".format(velocity, sample_rate), "w") as f:
            f.write(output_str)
        print("Finished CSV for velocity {0}, sample rate {1}".format(velocity, sample_rate))




#output_str = "X, IRI\n" #this is to make sure iris are computed properly
#for x in range(0, 100):
#    profile = road_profiles[x]
#    iri = profile.to_iri()
#    if(x < 33):
#
#        iris1.append(iri)
#    elif(x >= 33 and x <66):
#        iris2.append(iri)
#    else:
#        iris3.append(iri)
#    output_str += "{0}, {1}\n".format(x, iri)
#
#with open("/Users/gregoryislas/Documents/Mobilized/exp_iris.csv", "w") as f:
#    f.write(output_str)

#print("Mean, std dev, max, min for smooth profiles is {0}, {1}, {2}, {3}".format(np.mean(iris1), np.std(iris1), np.min(iris1), np.max(iris1)))
#print("Mean, std dev, max, min for rougher profiles is {0}, {1}, {2}, {3}".format(np.mean(iris2), np.std(iris2), np.min(iris2), np.max(iris2)))
#print("Mean, std dev, max, min for roughest profiles is {0}, {1}, {2}, {3}".format(np.mean(iris3), np.std(iris3), np.min(iris3), np.max(iris3)))

