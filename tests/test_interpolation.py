from quartercar.qc import QC
import matplotlib.pyplot as plt
from tests.make_profile import make_gaussian
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
num_profiles = 100


with open('interpolation_test.csv', 'w') as file:
    writer = csv.writer(file)
    header = ['Profile_id', 'Down-sampling', 'Interpolation', 'Down-sampling dx', 'MSE']
    writer.writerow(header)

    for s in range(num_profiles):

        sigma = 1  # in mm
        profile_len = 100 # in meters
        delta = .25 # original dx, in meters
        cutoff_freq = 0.15 #this is for the low-pass filter
        delta2 = .25 # final profile's dx, in meters
        seed = s
        final_dists, final_heights = make_gaussian(sigma, profile_len, delta, cutoff_freq, delta2, seed)[-2:]
        rp = RoadProfile(final_dists, final_heights)


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
                interp_dists, interp_accs = qc.inverse(down_accs, down_dists, velocities, interp_dx=dx, interp_type=i)[-2:]
                mse = mean_squared_error(accs, interp_accs)

                writer.writerow([s, d, i, down_dx, mse])
                if plots:
                    plt.figure(figsize=(20, 10))
                    plt.title('Profile no.: {0}, down-sampling: {1}, interpolation: {2}, down-sampling dx: {3}, MSE: {4}'.format(s, d, i, down_dx, mse))
                    plt.plot(new_distances, accs, 'go-', lw = .3, label = 'original')
                    plt.plot(down_dists, down_accs, 'o', mew = 7, label = 'downsampled')
                    plt.plot(interp_dists, interp_accs, '.-', label = 'interpolated', c = 'red')
                    plt.ylabel('Accelerations')
                    plt.xlabel('Distances')
                    plt.legend()
                    plt.show()
