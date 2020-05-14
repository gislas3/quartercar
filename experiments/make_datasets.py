import sys
sys.path.append("/home/annareisz/Documents/MC/quartercar")

from quartercar import qc, roadprofile
from tests import make_profile
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

import operator


def create_dataset(prof_dict):

    profile_list = []
    acceleration_list = []
    iri_list = []

    for prof_type in prof_dict.keys():

        n = prof_dict[prof_type]["num_profiles"]
        for i in range(n):

            dx = prof_dict[prof_type]['deltas'][i]
            length = prof_dict[prof_type]['lengths'][i]
            seed = prof_dict[prof_type]['seeds'][i]
            distances, elevations = make_profile.make_profile_from_psd(prof_type, 'sine', dx, length, seed)
            profile = roadprofile.RoadProfile(distances, elevations)
            profile_list.append(elevations)

            iri = profile.to_iri()
            iri_list.append(iri)

            car = qc.QC(208, 28, 1300, 18709, 127200)
            velocity = prof_dict[prof_type]['velocities'][i]
            sample_rate_hz = prof_dict[prof_type]['sample_rates'][i]
            T, yout, xout, new_distances, new_elevations = car.run(profile, dx, velocity, sample_rate_hz)
            accs = yout[:, -1]
            acceleration_list.append(accs)

    return profile_list, acceleration_list, iri_list


if __name__ == '__main__':

    length = 100
    sample_rate = 30
    for vel in range(1, 40):
    	print(vel)
    	N = 100
    	lengths = [length]*N
    	profiles = {
    		"A": {
    		"num_profiles": N,
    		"deltas": [.1]*N,
    		"lengths": lengths,
    		"seeds": np.arange(N),
    		"velocities": [vel]*N,
    		"sample_rates": [sample_rate]*N
    		},
    		"B": {
    		"num_profiles": N,
    		"deltas": [.1]*N,
    		"lengths": lengths,
    		"seeds": np.arange(N),
    		"velocities": [vel]*N,
    		"sample_rates": [sample_rate]*N
    		},
    		"C": {
    		"num_profiles": N,
    		"deltas": [.1]*N,
    		"lengths": lengths,
    		"seeds": np.arange(N),
    		"velocities": [vel]*N,
    		"sample_rates": [sample_rate]*N
    		}
    	}

    	rps, accs, iris = create_dataset(profiles)

    	iris = np.reshape(iris, newshape=(len(iris), 1))
    	"""
    	RP = [[] for i in range(5)]
    	for p in rps:
    		RP[0].append(np.mean(p))
    		RP[1].append(np.var(p))
    		RP[2].append(np.min(p))
    		RP[3].append(np.max(p))
    		RP[4].append(np.max(p)-np.min(p))

    	RP = np.asarray(RP).T

    	dataset = np.concatenate((RP, np.reshape(iris, newshape = (len(iris), 1))), axis = 1)
    	"""

    	ACC = [[] for i in range(5)]
    	for p in accs:
    		ACC[0].append(np.mean(p))
    		ACC[1].append(np.var(p))
    		ACC[2].append(np.min(p))
    		ACC[3].append(np.max(p))
    		ACC[4].append(np.max(p)-np.min(p))

    	ACC = np.asarray(ACC).T

    	dataset = np.concatenate((ACC, iris), axis = 1)

    	np.random.shuffle(dataset)

    	np.save("experiments/data3/predict_iri_data_len{}_vel{}".format(length, vel), dataset)













#
