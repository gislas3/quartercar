import gzip
import pickle
import numpy as np
import pandas as pd
import os
import datetime

from sklearn.linear_model import LinearRegression


class IriModel():
    def __init__(self, curve_length = 5, no_neighbors = 300, smoothing = 2):

        self.curve_length = curve_length
        #TODO: a way of estimating no_neighbors. Should depend on velocity and sample size
        self.no_neighbors = no_neighbors
        self.smoothing = smoothing

        #This get's filled up when training
        self.X = None
        self.iris = None
        self.neighbors = None
        self.neighbor_iris = None

    def _square_root_features(self, X):
        new = np.hstack((np.ones_like(X), X, np.sqrt(X)))
        return new

    def _get_neighbors(self):
        acc_var = self.X[:, 0][:, np.newaxis]
        acc_max = self.X[:, 1][:, np.newaxis]
        acc_var_root = self._square_root_features(acc_var)

        model = LinearRegression()
        model.fit(acc_var_root, acc_max)
        # TODO: variables should depend on X_train
        line_x = np.linspace(0, self.curve_length, self.no_neighbors)[:, np.newaxis]
        line_y = model.predict(self._square_root_features(line_x))
        curve = np.concatenate((line_x, line_y), axis = 1)
        self.neighbors = curve

    def _project_on_curve(self):
        # Finding closest neighbors
        closest_neighbors = np.zeros_like(self.X)
        for i in range(len(self.X)):
            min_distance = np.inf
            for neighbor in self.neighbors:
                distance = np.linalg.norm(neighbor-self.X[i])
                if distance < min_distance:
                    min_distance = distance
                    closest_neighbors[i] = neighbor

        # Making a default dict with empty lists
        curve_iri = {}
        for n in self.neighbors:
            curve_iri[tuple(n)] = []

        # The lists are filled up with the iri values of the original points
        for cn, iri in zip(closest_neighbors, self.iris):
            curve_iri[tuple(cn)].append(iri)

        # Taking the average of the lists
        # If there's an empty list it defaults to 0
        curve_points = []
        average_curve_iri = []
        for neighbor, irilist in curve_iri.items():
            curve_points.append(neighbor)
            if irilist == []:
                average_curve_iri.append(0)
            else:
                average_curve_iri.append(np.mean(irilist))
        self.neighbor_iris = average_curve_iri


    def _optimise_iri_curve(self):
        # Ordering points
        sorted_data = np.concatenate((self.neighbors, np.array(self.neighbor_iris)[:, np.newaxis]), axis = 1)
        sorted_data = np.array(sorted(sorted_data, key = lambda x: x[0]))

        # Interpolating
        for i in range(1, len(sorted_data)-1):
            if sorted_data[i][2] == 0:
                sorted_data[i][2] = sorted_data[i-1][2]

        #Smoothing
        for i in range(self.smoothing):
            for i in range(1, len(sorted_data)-1):
                sorted_data[i][2] = (sorted_data[i-1][2] + sorted_data[i +1][2])/2

        self.neighbors = sorted_data[:, :-1]
        self.neighbor_iris = sorted_data[:, -1]

    def fit(self, X, y):
        self.X = X
        self.iris = y
        self._get_neighbors()
        self._project_on_curve()
        self._optimise_iri_curve()

    def predict(self, test_set):
        predictions = np.zeros(len(test_set))
        for i in range(len(test_set)):
            min_distance = np.inf
            for j in range(len(self.neighbors)):
                distance = np.linalg.norm(self.neighbors[j]-test_set[i])
                if distance < min_distance:
                    min_distance = distance
                    predictions[i] = self.neighbor_iris[j]
        return predictions

"""
This is just a way to make the initial dataset

folder = 'NI_DATA'

def make_dataset(dir):
    count = 0
    filenames = os.listdir(dir)
    df = pd.DataFrame(columns = ['acc_var', 'acc_max', 'Velocity', 'Segment_ID', 'Timestamp' ])
    for filename in filenames:
        #print(filename)
        with gzip.open(dir+'/'+filename, 'rb') as f:
            trip = pickle.loads(f.read()) #each trip is a journey portion
            for i in range(len(trip)):
                segment = {'acc_var': np.var(trip[i]['Acc_Z']),
                            'acc_max': np.max(trip[i]['Acc_Z']-np.mean(trip[i]['Acc_Z'])),
                            'Velocity': trip[i]['Velocity'],
                            'Segment_ID': int(trip[i]['Segment_ID']),
                            'Timestamp': trip[i]['Timestamp']
                            }
                df = df.append(segment, ignore_index = True)
                count += 1
                #if count == 500:
                #    return df
    return df


all_segments = make_dataset(folder)
#print(len(all_segments))
#print(type(all_segments))

pickle_out = open("all_segments_no_mean.pickle","wb")
pickle.dump(all_segments, pickle_out)
pickle_out.close()
"""
