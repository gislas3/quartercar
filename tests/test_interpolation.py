from quartercar.qc import QC
import matplotlib.pyplot as plt
from tests.make_profile import make_gaussian, make_profile_from_psd
import numpy as np
from quartercar.roadprofile import RoadProfile
from sklearn.metrics import mean_squared_error
from scipy.interpolate import CubicSpline
import csv



def downsample(dists, accs, n_th = 2):
    """
    takes out every n_th acceleration measurement
    """

    num_of_samples = len(dists)//n_th + 1

    newdists = np.zeros(num_of_samples)
    newaccs = np.zeros(num_of_samples)

    for idx, i in enumerate(range(0, len(dists)-len(dists)%n_th, n_th)):
        newdists[idx] = dists[i]
        newaccs[idx] = accs[i]

    newdists[-1] = dists[-1]
    newaccs[-1] = accs[-1]

    return newaccs, newdists



plots = False
num_profiles_per_type = 50 #per profile_type
profile_types = ["A", "B", "C"]


with open('interpolation_test.csv', 'w') as file:
    writer = csv.writer(file)
    header = ['Profile_type', 'Profile_id', 'Down-sampling', 'Interpolation', 'Down-sampling dx', 'MSE_accs', 'RMSE_accs']
    writer.writerow(header)

    for p in profile_types:

        for s in range(num_profiles_per_type):

            seed = s
            profile_len = 100 # in meters
            delta = .25 # original dx, in meters
            final_dists, final_heights = make_profile_from_psd(p, 'sine', delta, profile_len, s)

            #print(len(final_heights))
            #sigma = 1  # in mm
            #cutoff_freq = 0.15 #this is for the low-pass filter
            #delta2 = .25 # final profile's dx, in meters
            #final_dists, final_heights = make_gaussian(sigma, profile_len, delta, cutoff_freq, delta2, seed)[-2:]

            rp = RoadProfile(final_dists, final_heights)

            #print(len(rp.get_elevations()))

            velocity = 30  # in m/s
            m_s, m_u, c_s, k_s, k_u = 208, 28, 1300, 18709, 127200
            qc = QC(m_s, m_u, c_s, k_s, k_u)
            sample_rate = 100
            dx = velocity/sample_rate
            _, yout, _, new_distances, _ = qc.run(rp, 100, velocity, sample_rate)
            accs = yout[:, -1]


            # d for "down sampling rate"
            for d in range(2, 6):
                down_accs, down_dists = downsample(new_distances, accs, n_th = d)
                down_dx = down_dists[1]-down_dists[0]
                velocities = [velocity]*len(down_dists)

                for i in ['linear',  'polinomial']:
                    #qc.py is changed such that the last two items it gives back are distances and accelerations
                    #interp_dists, interp_accs = qc.inverse(down_accs, down_dists, velocities, interp_dx=dx, interp_type=i)[-2:]
                    packed_vals = qc.inverse(down_accs, down_dists, velocities, interp_dx=dx, interp_type=i)
                    interp_elevs = packed_vals[0]
                    interp_dists = packed_vals[-2]
                    interp_accs = packed_vals[-1]

                    mse_accs = mean_squared_error(accs, interp_accs)
                    rmse_accs = np.sqrt(mse_accs)

                    #The two profiles' lenths are not the same... what's up with qc.run()?
                    #mse_profile = mean_squared_error(final_heights, interp_elevs)
                    #rmse_profile = np.sqrt(mse_profile)

                    writer.writerow([p, s, d, i, down_dx, mse_accs, rmse_accs])

                    if plots:
                        plt.figure(figsize=(20, 10))
                        plt.title('Profile type: {}, profile no: {}, down-sampling: {}, interpolation: {}, down-sampling dx: {}, MSE: {}, RMSE: {}'.format(p, s, d, i, down_dx, mse_accs, rmse_accs))
                        plt.plot(new_distances, accs, 'go-', lw = .3, label = 'original')
                        plt.plot(down_dists, down_accs, 'o', mew = 7, label = 'downsampled')
                        plt.plot(interp_dists, interp_accs, '.-', label = 'interpolated', c = 'red')
                        plt.ylabel('Accelerations')
                        plt.xlabel('Distances')
                        plt.legend()
                        plt.show()
