from quartercar import qc, roadprofile
from tests import make_profile
import numpy as np
import pandas as pd



class_dict = {}
class_types = ['A', 'B', 'C', 'D']
size_classes = 5000
seed = 1
len_array = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50])
dx_array = np.array([.25])
for cl in class_types:
    np.random.seed(seed)
    class_dict[cl] = {'Lengths': None, 'Dx': None}
    class_dict[cl]['Lengths'] = np.random.choice(len_array, size=size_classes, replace=True)
    class_dict[cl]['Dx'] = np.random.choice(dx_array, size=size_classes, replace=True)
    seed += 1

#now, generate profiles
seed = 1
data_dict = {'Profile': [], 'Dx': [], 'IRI': [], 'Length': [], 'Class': []}
for cl_type in class_types:
    for x in range(0, size_classes):
        len_x = class_dict[cl_type]['Lengths'][x]
        dx_x = class_dict[cl_type]['Dx'][x]
        orig_dists, orig_elevations = make_profile.make_profile_from_psd(cl_type, 'sine', dx_x, len_x, seed)
        profile = roadprofile.RoadProfile(orig_dists, orig_elevations)
        iri = profile.to_iri()
        data_dict['Profile'].append(orig_elevations)
        data_dict['Dx'].append(dx_x)
        data_dict['IRI'].append(iri)
        data_dict['Length'].append(len_x)
        data_dict['Class'].append(cl_type)
        seed += 1

final_df = pd.DataFrame(data=data_dict)
final_df.to_pickle('/Users/gregoryislas/Documents/Mobilized/data_dump/lstm_df1.pickle')








