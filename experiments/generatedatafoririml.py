import argparse
from quartercar import qc, roadprofile
from tests import make_profile
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description='Program for generating data used to build ML models to predict IRI and/or road profile')

parser.add_argument('-A', '--A_profiles', '--num_A', nargs='?', type=int, default=0, const=0, metavar='N',
                        help='The number of class A road profiles to generate')
parser.add_argument('-B', '--B_profiles', '--num_B', nargs='?', type=int, default=0, const=0, metavar='N',
                        help='The number of class B road profiles to generate')
parser.add_argument('-C', '--C_profiles', '--num_C', nargs='?', type=int, default=0, const=0, metavar='N',
                        help='The number of class C road profiles to generate')
parser.add_argument('-D', '--D_profiles', '--num_D', nargs='?', type=int, default=0, const=0, metavar='N',
                        help='The number of class D road profiles to generate')
parser.add_argument('-E', '--E_profiles', '--num_E', nargs='?', type=int, default=0, const=0, metavar='N',
                        help='The number of class E road profiles to generate')
parser.add_argument('-F', '--F_profiles', '--num_F', nargs='?', type=int, default=0, const=0, metavar='N',
                        help='The number of class F road profiles to generate')
parser.add_argument('-G', '--G_profiles', '--num_G', nargs='?', type=int, default=0, const=0, metavar='N',
                        help='The number of class G road profiles to generate')
parser.add_argument('-H', '--H_profiles', '--num_H', nargs='?', type=int, default=0, const=0, metavar='N',
                        help='The number of class H road profiles to generate')
parser.add_argument('--m_s', '--sprung_mass', nargs='?', type=float, default=240, const=240, metavar='X.X',
                        help='The sprung mass (in kg) of the vehicle in simulation')
parser.add_argument('--m_u', '--unsprung_mass', nargs='?', type=float, default=36, const=36, metavar='X.X',
                        help='The unsprung mass (in kg) of the vehicle to use in simulation')
parser.add_argument('--c_s', '--damping', '--damping_coefficient', nargs='?', type=float, default=980, const=980,
                    metavar='X.X', help='The damping constant (in N*s/m) of the vehicle to use in simulation ')
parser.add_argument('--k_s', '--suspension', '--suspension_spring_rate', nargs='?', type=float, default=16000,
                    const=16000, metavar='X.X',
                    help='The sprung mass spring constant (in N/m) of the vehicle to use in simulation')
parser.add_argument('--k_u', '--tire', '--tire_spring_rate', nargs='?', type=float, default=160000,
                    const=160000, metavar='X.X',
                    help='The unsprung mass spring constant (in N/m) of the vehicle to use in simulation')
parser.add_argument('--interp', '--interpolation', nargs='?', default='None', const='None', metavar='XXXX',
                        help='The type of interpolation to use when interpolating the accelerations')
parser.add_argument('--path', '--output_path', nargs='?', default='None', const='None', metavar='XXXX',
                        help='The output path to save the data')

parser.add_argument('--sample_rate', '--sr', '--sr_hz', nargs='?', type=int, default=100, const=100,
                    metavar='N', help='The sample rate of the accelerometer as the car is driving over the road profile')

parser.add_argument('--dx', '--sample_spacing', nargs='?', type=float, default=.1, const=.1,
                    metavar='X.X', help='The spacing between samples (in m) when the car drives over the profile')
parser.add_argument('--velocity', '--vel', nargs='?', type=float, default=5, const=5,
                    metavar='X.X', help='The velocity of the car (in m/s) to use for the simulation')

parser.add_argument('--ds', '--downsample', '--down_sample_rate', nargs='?', type=float, default=.2, const=.2,
                    metavar='X.X', help='The spacing between samples (in m) after downsampling')

parser.add_argument('--seed', '--random_seed', nargs='?', type=int, default=55, const=55,
                    metavar='N', help='The random seed to use so results can be reproduced')

parser.add_argument('--format', '--write_to_file', nargs='?', type=int, default=1, const=1,
                    metavar='N', help='Flag denoting whether or not to write the outputs to a file')

parser.add_argument('-l', '--length', '--profile_length', nargs='?', type=float, default=100, const=100,
                    metavar='N', help='The length (in m) of the profiles to generate')

parser.add_argument()



args = parser.parse_args()

# We want to run the simulation for each road profile, generate a csv file with the acceleration of the
# sprung mass, the road profile as input, the IRI, the velocity, and the sample spacing

class_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
num_classes = [args.A_profiles, args.B_profiles, args.C_profiles, args.D_profiles,
               args.E_profiles, args.F_profiles, args.G_profiles, args.H_profiles]
vehicle = qc.QC(m_s=args.m_s, m_u=args.m_u, c_s=args.c_s, k_s=args.c_s, k_u=args.k_u)
seed = 1
for t, n in zip(class_types, num_classes):
    for x in range(0, n):
        orig_dists, orig_elevations = make_profile.make_profile_from_psd(t, 'sine', .01, args.length, seed)
        profile = roadprofile.RoadProfile(orig_dists, orig_elevations)
        iri = profile.to_iri()
        T, yout, xout, dists, elevations = vehicle.run(profile, 100, args.velocity, args.sample_rate)
        accs = yout[:, -1]
        dx_sample = dists[2] - dists[1]
        df = pd.DataFrame({'Sprung_Mass_Acc': accs, 'Profile_Sample': elevations, 'IRI': len(accs)*[iri],
                           'Velocity(m/s)': len(accs)*[args.velocity], 'DX': len(accs)*[dx_sample]})
        df.to_csv('{0}/{1}_{2}_{3}.csv'.format(args.path, t, x, args.velocity), index=False)
        seed += 1






