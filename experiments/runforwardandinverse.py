import argparse
from quartercar import qc, roadprofile
from tests import make_profile
import numpy as np
import pandas as pd

#What do we want...

#We want a parameter to generate X road profiles of class 'A', 'B', 'C', or 'D'

#We want a parameter(s) for vehicle parameters

#We want to specify interpolation method

#We want to specify output files location?

#We want to specify sample rate/sample spacing

#We want to specify downsampling rate

#We want to specify random seed

#We want to specify length of road profiles

#Eventually, want to specify split of road profiles, and sample rate for each one

def make_road_profile_list(prof_type, num_profiles, dx, length, seed, flag, path):
    list_profiles = []
    for x in range(0, num_profiles):
        distances, elevations = make_profile.make_profile_from_psd(prof_type, 'sine', dx/2, length, seed)
        profile = roadprofile.RoadProfile(distances, elevations)
        list_profiles.append(profile)
        if flag:
            df = pd.DataFrame(data={"distances": distances, "elevations": elevations})
            df.to_csv("{0}/{1}_{2}_profile".format(path, prof_type, x), index=False)
    return list_profiles

def run_simulations(prof_type, list_profiles, car, dx, velocity, sample_rate_hz, flag, path):
    list_accs = []
    for p in list_profiles:
        T, yout, xout, new_distances, new_elevations = car.run(p, dx, velocity, sample_rate_hz)
        accs = yout[:, -1]
        if flag:
            df = pd.DataFrame(data={"distances": new_distances, "accelerations": accs})
            df.to_csv("{0}/{1}_{2}_accelerations".format(path, prof_type, x), index=False)
        list_accs.append((new_distances, accs))
    return list_accs

def downsample(list_accs, dx):
    downsampled_accs = []
    for x in range(0, len(list_accs)):
        new_dists, new_accs = [0], [list_accs[x][1][0]] #should always start at 0
        dists, accs = list_accs[x][0], list_accs[x][1]
        curr_dist = dx
        for y in range(0, len(dists)):
            d = dists[y]
            acc = accs[y]
            if d >= curr_dist: #Not assuming that dx is a multiple of the sampling interval
                new_dists.append(curr_dist)
                new_accs.append(acc)
                curr_dist += dx
        downsampled_accs.append((np.array(new_dists), np.array(new_accs)))

    return downsampled_accs

def run_inverses(prof_type, ds_list, car, velocity, sample_rate, dx, interp, flag, path):
    inverse_list = []
    for x in range(0, len(ds_list)):
        dists, accs = ds_list[x][0], ds_list[x][1]
        packed_vals = car.inverse(accs, dists, [velocity]*len(dists), sample_rate, dx, interp)
        #inverse_list.append(())
        if flag:
            new_dists = np.arange(0, dists[-1], dx)
            #print("new dists is {0}".format(new_dists))
            #print("len new dists is {0}, elevations is {1}".format(len(new_dists), len(packed_vals[0])))
            df_inv_profile = pd.DataFrame(data={'distances': new_dists, "elevations": packed_vals[0]})
            df_inv_profile.to_csv("{0}/{1}_{2}_{3}_interp_profile".format(path, prof_type, x, interp), index=False)
            df_inv_accs = pd.DataFrame(data={'distances': new_dists, "accelerations": packed_vals[-1]})
            df_inv_accs.to_csv("{0}/{1}_{2}_{3}_interp_accs".format(path, prof_type, x, interp), index=False)




parser = argparse.ArgumentParser(description='Program for running forward and inverse methods for the qc, and outputting accelerations')

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

parser.add_argument('--dx', '--sample_spacing', nargs='?', type=float, default=.1, const=.1,
                    metavar='X.X', help='The spacing between samples (in m) when the car drives over the profile')
parser.add_argument('--vel', '--velocity', nargs='?', type=float, default=5, const=5,
                    metavar='X.X', help='The velocity of the car (in m/s) to use for the simulation')

parser.add_argument('--ds', '--downsample', '--down_sample_rate', nargs='?', type=float, default=.2, const=.2,
                    metavar='X.X', help='The spacing between samples (in m) after downsampling')

parser.add_argument('--seed', '--random_seed', nargs='?', type=int, default=55, const=55,
                    metavar='N', help='The random seed to use so results can be reproduced')

parser.add_argument('--write', '--write_to_file', nargs='?', type=int, default=1, const=1,
                    metavar='N', help='Flag denoting whether or not to write the outputs to a file')

parser.add_argument('-l', '--length', '--profile_length', nargs='?', type=float, default=100, const=100,
                    metavar='N', help='The length (in m) of the profiles to generate')



#parser.add_argument('--file', nargs='*', metavar='FILE_NAME',
#                        help='The bytes file(s) to read the data stream from')

args = parser.parse_args()

#Step 1: Create the vehicle to use for the simulations
vehicle = qc.QC(m_s=args.m_s, m_u=args.m_u, c_s=args.c_s, k_s=args.c_s, k_u=args.k_u)

#Step 2: Create the road profiles
profile_types  = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
profile_nums = [args.A_profiles, args.B_profiles, args.C_profiles, args.D_profiles, args.E_profiles, args.F_profiles,
                args.G_profiles, args.H_profiles]
prof_dict = {}
for x in range(0, len(profile_types)):
    t = profile_types[x]
    n = profile_nums[x]
    prof_dict[t] = make_road_profile_list(t, n, args.dx, args.length, args.seed, args.write, args.path)



#Step 3: Compute the sampling rate
sample_rate = args.vel/args.dx
acc_dict = {}
#Step 4: Run the forward simulation for all the profiles, get the accelerations
for x in range(0, len(profile_types)):
    t = profile_types[x]
    list_profiles = prof_dict[t]
    acc_dict[t] = run_simulations(t, list_profiles, vehicle, args.dx, args.vel, sample_rate, args.write, args.path)

down_dict = {}
#Step 5: Down sample the accelerations
for x in range(0, len(profile_types)):
    t = profile_types[x]
    down_dict[t] = downsample(acc_dict[t], args.ds)

inverse_dict = {}
#Step 6: Compute the inverse
for x in range(0, len(profile_types)):
    t = profile_types[x]
    downsampled_list = down_dict[t]
    run_inverses(t, downsampled_list, vehicle, args.vel, sample_rate, args.dx, args.interp, args.write, args.path)



