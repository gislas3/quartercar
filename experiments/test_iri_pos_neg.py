import argparse
from quartercar import qc, roadprofile
from tests import make_profile
import numpy as np
import pandas as pd
from scipy.linalg import expm, inv
import sys


for x in range(0, 10):
    orig_dists, orig_elevations = make_profile.make_profile_from_psd('A', 'sine', .01, 100, x)
    profile = roadprofile.RoadProfile(orig_dists, orig_elevations)
    true_iri = profile.to_iri()
    profile2 = roadprofile.RoadProfile(orig_dists, -1*orig_elevations)
    iri2 = profile2.to_iri()
    print(true_iri)
    print(iri2)
    print('')