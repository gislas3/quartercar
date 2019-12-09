from quartercar import qc, roadprofile
from tests import make_profile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

import operator


def create_dataset(prof_dict):
    """
    makes any amount of profiles of any type
    ex. prof dict = {"A": {num_profiles: N, deltas: [int]*N, lengths: [int]*N, seeds: [int]*N, velocities: [int]*N, sample_rates: [int]*N}}
    """
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


N = 100 #let's make a 100 profiles per profile type
lengths = np.arange(1, N+1)*50
profiles = {
    "A": {
    "num_profiles": N,
    "deltas": [.1]*N,
    "lengths": lengths,
    "seeds": np.arange(N),
    "velocities": [30]*N,
    "sample_rates": [100]*N
    },
    "B": {
    "num_profiles": N,
    "deltas": [.1]*N,
    "lengths": lengths,
    "seeds": np.arange(N),
    "velocities": [30]*N,
    "sample_rates": [100]*N
    },
    "C": {
    "num_profiles": N,
    "deltas": [.1]*N,
    "lengths": lengths,
    "seeds": np.arange(N),
    "velocities": [30]*N,
    "sample_rates": [100]*N
    },
    "D": {
    "num_profiles": N,
    "deltas": [.1]*N,
    "lengths": lengths,
    "seeds": np.arange(N),
    "velocities": [30]*N,
    "sample_rates": [100]*N
    }
}

rps, accs, iris = create_dataset(profiles)

iris = np.reshape(iris, newshape=(len(iris), 1))

RP = [[] for i in range(5)]
for p in rps:
    RP[0].append(np.mean(p))
    RP[1].append(np.var(p))
    RP[2].append(np.min(p))
    RP[3].append(np.max(p))
    RP[4].append(np.max(p)-np.min(p))

RP = np.asarray(RP).T

dataset = np.concatenate((RP, np.reshape(iris, newshape = (len(iris), 1))), axis = 1)

ACC = [[] for i in range(5)]
for p in accs:
    ACC[0].append(np.mean(p))
    ACC[1].append(np.var(p))
    ACC[2].append(np.min(p))
    ACC[3].append(np.max(p))
    ACC[4].append(np.max(p)-np.min(p))

ACC = np.asarray(ACC).T

dataset = np.concatenate((ACC, dataset), axis = 1)

np.random.shuffle(dataset)

np.save("predict_iri_data", dataset)













#
