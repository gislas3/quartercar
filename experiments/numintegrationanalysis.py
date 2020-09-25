import numpy as np
from matplotlib import pyplot as plt
#import filterpy
from experiments import computeisoroughness as cisr
from quartercar import roadprofile as rp
from quartercar import qc
from tests import make_profile as mp
from scipy.signal import periodogram, welch, detrend, butter, sosfilt
from scipy.integrate import simps, cumtrapz
from scipy.stats import norm as st_norm
import sys, time, pickle
import pandas as pd
from numpy.random import RandomState


def generate_road_profiles(profile_len, num_profiles, prof_types, dx=.01, start_seed=1):
    #num_segs = 1
    #assume seglens and profile len add up to each other
    #if not np.isclose(np.sum(seg_lens), profile_len):
    #    raise ValueError("Please enter segment lengths that add up to the profile length that you want")
    seed = start_seed
    full_st_time = time.time()
    for ptype in prof_types:
        class_st_time = time.time()
        for x in range(0, num_profiles):
            st_time = time.time()
            distances, elevations, true_gn0 = mp.make_profile_from_psd(ptype, 'sine', dx, profile_len, seed=seed, ret_gn0=True)
            plen_fmt = profile_len#/1000
            np.save('/Users/gregoryislas/Documents/Mobilized/data_dump/Road_Profiles/Profile_{0}_{1}_{2}_distances'.format(ptype, plen_fmt, x), distances)
            np.save('/Users/gregoryislas/Documents/Mobilized/data_dump/Road_Profiles/Profile_{0}_{1}_{2}_elevations'.format(ptype, plen_fmt, x), elevations)
            np.save('/Users/gregoryislas/Documents/Mobilized/data_dump/Road_Profiles/Profile_{0}_{1}_{2}_gn0'.format(ptype, plen_fmt, x), np.array(true_gn0))
            seed += 1
            print("Finsished saving Profile_{0}_{1}_{2}, took {3} seconds".format(ptype, plen_fmt, x, time.time() - st_time))
        print("Took {0} seconds for {1} profiles of class {2}, len {3}\n\n".format(time.time() - class_st_time, num_profiles, ptype, plen_fmt))
    print("Total runtime was {0} seconds".format(time.time()- full_st_time))


def get_car_list():
    #car1 = qc.QC(m_s=380, m_u=55, k_s=20000, k_u=350000, c_s=2000)
    #car2 = qc.QC(m_s=380, m_u=55, k_s=60000, k_u=350000, c_s=8000)
    car1 = qc.QC(m_s=240, m_u=36, k_s=16000, c_s=.5*980, k_u=160000)
    #car2 = qc.QC(m_s=240, m_u=36, k_s=16000, c_s=980, k_u=160000)
    #car3 = qc.QC(m_s=240, m_u=36, k_s=16000, c_s=2*980, k_u=160000)
    #car4 = qc.QC(m_s=240, m_u=36, k_s=16000, c_s=3*980, k_u=160000)
    car2 = qc.QC(m_s=250, m_u=40, k_s=28000, c_s=2000, k_u=125000)
    car3 = qc.QC(m_s=208, m_u=28, k_s=18709, c_s=1300, k_u=127200)
    car4 = qc.QC(m_s=300, m_u=50, k_s=18000, c_s=1200, k_u=180000)
    car5 = qc.QC(m_s=243, m_u=40, k_s=14671, c_s=370, k_u=124660)
    car6 = qc.QC(m_s=257, m_u=31, k_s=13100, c_s=400, k_u=126000)
    car7 = qc.QC(m_s=290, m_u=59, k_s=16812, c_s=1000, k_u=190000)
    suv1 = qc.QC(m_s=650, m_u=55, k_s=27500, c_s=3000, k_u=237000)
    suv2 = qc.QC(m_s=737.5, m_u=62.5, k_s=26750, c_s=3500, k_u=290000)
    bus = qc.QC(m_s=4500, m_u=500, k_s=300000, c_s=20000, k_u=1600000)
    return [('car1', car1), ('car2', car2), ('car3', car3),  ('car4', car4), 
    ('car5', car5), ('car6', car6), ('car7', car7), ('suv1', suv1), ('suv2', suv2), ('bus', bus)]

def simulate_driving_over_profiles(num_profiles, prof_types, velocities, sample_rates, 
noise_sigmas, cutoff_freqs, filter_orders, start_seed=1, direc='/Users/gregoryislas/Documents/Mobilized/data_dump/Road_Profiles/',
fname='/Users/gregoryislas/Documents/Mobilized/data_dump/kalman_R_exp'):
    data_dict = {'Profile_Type': [], 'Profile_Number': [], 
    'GN0': [], 'Car_Name': [], 'Velocity': [],  'Noise_Sigma': [], 
    'Cutoff_Freq': [], 'Cutoff_Wavelen': [], 'Filter_Order': [], 'Mean_Error_Sprung_V': [],
    'Var_Error_Sprung_V': [], 'Mean_Error_Sprung_D': [], 'Var_Error_Sprung_D': [],
    'Cov_Error_Acc_Vel': [], 'Cov_Error_Acc_Disp': [], 'Cov_Error_Vel_Disp': [],
    'Sample_Rate_Hz': [], 'Random_Seed': []}
    seed = start_seed
    #rng = default_rng()
    prog_st_time = time.time()
    for ptype in prof_types:
        for n in range(0, num_profiles):
            distances = np.load("{0}Profile_{1}_{2}_distances.npy".format(direc, ptype, n))
            elevations = np.load("{0}Profile_{1}_{2}_elevations.npy".format(direc, ptype, n))
            gn0 =  np.load("{0}Profile_{1}_{2}_gn0.npy".format(direc, ptype, n))
            profile = rp.RoadProfile(distances=distances, elevations=elevations)
            st_time = time.time()
            print("Starting simulation for profile {0}, {1}".format(ptype, n))
            for c in get_car_list():
                cname, vehicle = c[0], c[1]
                st_vtime = time.time()
                for v in velocities:
                    for sr in sample_rates:
                        acc_times = []
                        T, yout, xout, new_dists, new_els, vs = vehicle.run2(profile, acc_times, v0=v, final_sample_rate=sr)
                        #remove the first 50 entries to account for the random error
                        times = T[50:]
                        true_accs = yout[50:, -1]
                        true_vels = xout[50:, 1]
                        true_disps = xout[50:, 0]
                        # plt.plot(true_vels, label='True_vels')
                        # plt.plot(detrend(cumtrapz(true_accs, times, initial=0)), label='Int Vels')
                        # plt.legend()
                        # plt.show()
                        # plt.plot(true_disps, label='True_disps')
                        # plt.plot(detrend(cumtrapz(detrend(cumtrapz(true_accs, times, initial=0)), times, initial=0)), label='Int Disps')
                        # plt.legend()
                        # plt.show()
                        # sys.exit(0)
                        sr_time = time.time()
                        for sig in noise_sigmas:
                            rs = RandomState(seed)
                            noise = rs.normal(loc=0, scale=np.sqrt(sig))
                            acc_noise = true_accs + noise
                            for wave_c in cutoff_freqs:
                            #wave_c = 100 #cutoff freq, spatial domain
                                time_cutoff_freq = 1/wave_c * v # 
                                for order in filter_orders:
                                    sos = butter(order, time_cutoff_freq, btype='highpass', analog=False, output='sos', fs=sr)
                                    est_vels = sosfilt(sos, cumtrapz(acc_noise, times, initial=0))
                                    #vs_non_noise = cumtrapz(accs,times, )
                                    est_disps = sosfilt(sos, cumtrapz(est_vels, times, initial=0))
                                    er_accs = true_accs - acc_noise
                                    er_vels = true_vels - est_vels
                                    er_disps = true_disps - est_disps
                                    mean_e_sv = np.mean(er_vels)
                                    var_e_sv = np.var(er_vels)
                                    mean_e_disps = np.mean(er_disps)
                                    var_e_disps = np.var(er_disps)
                                    cov_mat = np.cov(np.vstack([er_accs, er_vels, er_disps]))
                                    #print(np.vstack([er_accs, er_vels, er_disps]).shape)
                                    #print(cov_mat.shape)
                                    #sys.exit(1)
                                    data_dict['Profile_Type'].append(ptype)
                                    data_dict['Profile_Number'].append(n)
                                    data_dict['GN0'].append(gn0)
                                    data_dict['Car_Name'].append(cname)
                                    data_dict['Velocity'].append(v)
                                    data_dict['Noise_Sigma'].append(sig)
                                    data_dict['Cutoff_Freq'].append(time_cutoff_freq)
                                    data_dict['Cutoff_Wavelen'].append(wave_c)
                                    data_dict['Filter_Order'].append(order)
                                    data_dict['Mean_Error_Sprung_V'].append(mean_e_sv)
                                    data_dict['Var_Error_Sprung_V'].append(var_e_sv)
                                    data_dict['Mean_Error_Sprung_D'].append(mean_e_disps)
                                    data_dict['Var_Error_Sprung_D'].append(var_e_disps)
                                    data_dict['Cov_Error_Acc_Vel'].append(cov_mat[0][1])
                                    data_dict['Cov_Error_Acc_Disp'].append(cov_mat[0][2])
                                    data_dict['Cov_Error_Vel_Disp'].append(cov_mat[1][2])
                                    data_dict['Sample_Rate_Hz'].append(sr)
                                    data_dict['Random_Seed'].append(seed)
                                    
                            seed += 1
                        print("Finished round for one sample rate, time was {0} seconds".format(time.time() - sr_time))
                    print("Finished round for one velocity, time was {0} seconds".format(time.time() - st_vtime))
                print("Finished round for one profile, time was {0} seconds\n\n".format(time.time() - st_time))
    df = pd.DataFrame(data_dict)
    try:
        with open('{0}.pickle'.format(fname), 'wb') as f:
            pickle.dump(df, f)
    except Exception as e:
        print("Failed saving dataframe as pickle file, saving as csv")
        df.to_csv('{0}.csv'.format(fname), index=False)
    print("Total program time was {0} seconds".format(time.time() - prog_st_time))


for plen, start_seed in zip([500, 1000, 2000, 5000, 10000], [1, 500, 1500, 30000, 6900]):
    generate_road_profiles(plen, 30, ['A', 'B', 'C'], dx=.01, start_seed=start_seed)

#simulate_driving_over_profiles(30, ['A', 'B', 'C'], [5, 10, 15, 20, 25, 30], [100, 200], [.1, .05, .01, .005, .001,], [100, 150, 200, 250, 300, 350, 400, 450, 500], list(range(1, 11)), start_seed=1, direc='/Users/gregoryislas/Documents/Mobilized/data_dump/Road_Profiles/')

#simulate_driving_over_profiles(30, ['A', 'B', 'C'], [5, 10, 15, 20, 25, 30], [100, 200], [.1, .05, .01, .005, .001,], 
#[200], [6], start_seed=1, direc='/Users/gregoryislas/Documents/Mobilized/data_dump/Road_Profiles/', fname='/Users/gregoryislas/Documents/Mobilized/data_dump/kalman_R_exp_cars')
