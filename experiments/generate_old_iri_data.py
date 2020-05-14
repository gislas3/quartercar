import argparse
from quartercar import qc, roadprofile
from tests import make_profile
import numpy as np
import pandas as pd
from scipy.linalg import expm, inv
import sys



def compute_slopes(init_slope, accs, vels, deltas, l=30, centered=False):
    """
    Computes slopes from a list accelerometer measurements, velocities, and spaces between samples
    :param init_slope: The slope to use as the estimated initial values
    :param accs: The list of accelerometer measurements (assumes units G's)
    :param vels: The list of velocities in m/s
    :param deltas: The list of space between consecutive measurments
    :param l: The longest wavelength of interest of the profile (default: 30 m)
    :param centered: Whether or not the mean has already been removed from the accelerometer measurements (default: False)
    :return: A list of slope profile values, or the first order integration of the accelerometer series
    """
    slopes = np.zeros(len(accs))
    slopes[0] = init_slope
    #deltas = np.zeros(len(accs))
    #accs2 = convert_acc(accs, True)
    accs2 = np.copy(accs)
    mn = 0
    if(not centered):
        mn = np.mean(accs2)
    #print("Mean is {0}".format(mn))
    #plt.plot(accs2, 'r')
    #plt.plot(np.array(accs2) - mn, 'b')
    #plt.show()
    vel = np.mean(vels)
    #print("Inside compute slopes, vels is {0}".format(vels))
    prev_slope = init_slope
    for x in range(1, len(slopes)):
        delta = deltas[x]
        #vel = vels[x]
        #delta = vel / sr  # distanace traveled = m/s/(number of samples per second)
        C = delta / (3 * l)  # constant related to 3 * longest wavelength of interest
        #deltas[x] = delta
        #slopes[x] = C * slopes[x - 1] + delta * ((accs2[x] - mn) / vel ** 2)  # slope = C*previous slope + distance *(acceleration - mean)/velocity**2
        slopes[x] = C * slopes[x-1] + delta * ((accs2[x] - mn) / vel ** 2)
        #try:
        #    print("OH SHIT, len(slopes[x])) is {0}, x is {1}, slopes[x] is {2}".format(len(slopes[x]), x, slopes[x]))
        #except Exception as e:
        #    pass

    #if(len(deltas) > 1):
    #    deltas[0] = deltas[1]
    #print("Shape slopes is {0}".format(slopes.shape))
    return slopes#, deltas

def compute_iri_from_slopes(slopes, intervals, vels, initial_value=None): #disps=None):
    """
    This function returns the IRI from the calculated slope/displacements of a road profile
    the formula is taken from On the Calculation of International Roughness Index from Longitudinal Road Profile, by Michael Sayers
    and computation of the profile from the acclerometer is taken from Guidelines for Longitudinal Pavement Profile Measurement
    by S. M. Karamihas and T. D. Gillespie
    :param slopes: An array of the estimated slopes of the road profile at each point in space
    :param intervals: An array of distances from the beginning of the measurement process
    :param vels: An array of velocities that the car was traveling during the measurement
    :return: The IRI value of the road
    """

    # parameters taken from Sayers paper, correspond to quarter car simluation parameters
    c = 6.0
    k_1 = 653
    k_2 = 63.3
    mu = 0.15
    # print(disps)
    # smoothed_hts = get_smooth_hts(disps, intervals)
    # print(smoothed_hts)
    # print(slopes)
    #print("orig slopes 0 is {0}".format(slopes[0]))
   # if(disps is None):

    #else:
    #smoothed_slps, dist_covered = get_smooth_slps(disps, intervals)
    #print("Smoothed slps 0 is {0}".format(smoothed_slps[0]))
    # the following parameters are also taken from the Sayers paper
    a_mat = np.array([[0, 1, 0, 0], [-k_2, -c, k_2, c], [0, 0, 0, 1], [k_2 / mu, c / mu, -(k_1 + k_2) / mu, -c / mu]])
    inv_a = inv(a_mat)
    b_mat = np.array([[0], [0], [0], [k_1 / mu]])
    vel = 80/3.6 #convert simulated speed of 80 km/hr to m/s
    curr_pro_len = 0  # keep track of the estimated distance traveled along the profile
    # TODO: optimize calculation to not use matrix math
    # C = np.array([[1, 0, -1, 0]])
    #iri_dict = {}  # store the computed iri values
    curr_sum = 0  # current numerator for the calculation
    n = 0  # current denominator for the calculation
    #prev_ind = 0  # previous height index
    # v_list = []
    #prev_seg = None  # keep track of previous segment
    #mean_vel = np.mean(vels)
    #mean_delta = np.mean(intervals)
    #max_delta = np.max(intervals)

    #
    #v_thresh = 5 #must be going at least 5 m/s on average to use displacements from vehicle body
    #mn_delta_thresh = .3#average sample rate must be <= 300 mm  - tweak this as needed
    #max_delta_thresh = .6 #if there are any greater than or equal to 600 mm, consider invalid - tweak as needed
    #logging.info("Mean vel is {0}, mean delta is {1}, max delta is {2}, stddev delta is {3}".format(mean_vel, mean_delta, max_delta, np.std(intervals)))
    #if(mean_vel >= v_thresh and mean_delta <= mn_delta_thresh and max_delta <= max_delta_thresh):
        #smoothed_slps, dist_covered = get_smooth_slopes(slopes, intervals)
    smoothed_slps = slopes #Not applying filter since already filtered
    delta = intervals[0]
    ex_a = expm(a_mat * delta / vel) #just use average interval as sample rate
    term2 = np.matmul(np.matmul(inv_a, (ex_a - np.eye(4))), b_mat)
    for ind in range(0, len(smoothed_slps)):  # again, for more details, see Sayers paper
        slp = smoothed_slps[ind]
        intv = intervals[ind]
            #print("Slp is {0}".format(slp))
            #vel = vels[ind]
            #seg = segs[ind]

            #if (seg != prev_seg):  # if at a new segment, compute the IRI
            #    if (prev_seg != None):
            #        prev_seg_len = seg_len_dict[prev_seg]
            #        if (curr_pro_len / prev_seg_len >= .7):  # cover at least 70% of the profile to be considered
            #            avg_t = average_timestamp(times[prev_ind:ind])
            #            avg_v = np.average(vels[prev_ind:ind])
            #            if (prev_seg in iri_dict):
            #                iri_dict[prev_seg].append((curr_sum / n, min(curr_pro_len / prev_seg_len, 1), avg_t, avg_v))
            #            else:
            #                iri_dict[prev_seg] = [(curr_sum / n, min(curr_pro_len / prev_seg_len, 1), avg_t, avg_v)]
            #            # print("IRI is {0}".format(curr_sum/n))
            #            # if(curr_pro_len > prev_seg_len):#for testing only
            #            #    print("Curr pro len is {0}".format(curr_pro_len))
            #            #    print("Prev seg len is {0}".format(prev_seg_len))
            #        curr_pro_len, curr_sum, n = 0, 0, 0
            #        prev_ind = ind
        if(ind == 0):
            x = initial_value
                #if initial_value is None:
                #    dist, next_ind = intv, 1
                #    #print("Smoothed slopes 0 is {0}".format(smoothed_slps[0]))
                #    sum41 = intv*smoothed_slps[ind]
                #    while(next_ind < len(smoothed_slps) and dist < 11):
                #        #print("smoothed slopes is {0}".format(smoothed_slps[next_ind]))
                #        #print("Intervals is {0}".format(intervals[next_ind]))
                #        sum41 += intervals[next_ind]*smoothed_slps[next_ind]
                #        dist += intervals[next_ind]
                #        next_ind += 1
#
                ##next_ind = min(int(np.round(ind + 11 / intv)), len(smoothed_slps) - 1)  # get the next index
                ##h1 = disps[ind]
                ##h2 = disps[next_ind]
#
                ## print("Next ind is {0}".format(next_ind))
                ## print("denom is {0}".format(intv*next_ind))
                ##x = np.array([[(h2 - h1) / (intv * next_ind)], [0], [(h2 - h1) / (intv * next_ind)], [0]])
                ##print("Curr sum is {0}".format(curr_sum))
                ##print("Dist is {0}".format(dist))
                #    x  = np.array([[sum41/11], [0], [sum41/11], [0]])
                ##print("x is {0}".format(x))
                ## prev_ind = ind
                #else:
                #    x = initial_value
            #else:  # see Sayers paper
                 #SHIT - this should be intv/(80/3.6) (if doing m/s)
                # print("DT is {0}".format(intv/vel))
                # ex_a = np.exp(a_mat*intv/vel)
        term1 = np.matmul(ex_a, x)

                # print(term1)
                # print(term2)
        x = term1 + term2 * slp
                # print(slp)
                # print()
        curr_sum += abs(x[0, 0] - x[2, 0])
        n += 1
        curr_pro_len += intv
            # print(x)

            # print("X is {0}".format(ind/4))
            # print("IRI is {0}".format(curr_sum/n))
            # print("Disp is {0}".format(curr_sum/n * ind/4))
            # iris.append(curr_sum/n)
            # print("")

            # print(intv)
            # prev_ind = ind
        #    prev_seg = seg
        #if len(segs) >= 2 and segs[-1] == segs[-2]:  # if at end and still have to compute an IRI value
        #    prev_seg_len = seg_len_dict[prev_seg]
        #    if (curr_pro_len / prev_seg_len >= .7):  # cover at least 70% of the profile to be considered
        #        avg_t = average_timestamp(times[prev_ind:ind])
        #        avg_v = np.average(vels[prev_ind:ind])
        #        if (prev_seg in iri_dict):
        #            iri_dict[prev_seg].append((curr_sum / n, min(curr_pro_len / prev_seg_len, 1), avg_t, avg_v))
        #        else:
        #            iri_dict[prev_seg] = [(curr_sum / n, min(curr_pro_len / prev_seg_len, 1), avg_t, avg_v)]
        #        # if (curr_pro_len > prev_seg_len):  # for testing only
        #        #    print("Curr pro len is {0}".format(curr_pro_len))
        #        #    print("Prev seg len is {0}".format(prev_seg_len))
        #    # v_list.append(np.average(vels[prev_ind:]))
    to_ret_iri = None
    if(n > 0 and curr_sum >= 0):
        to_ret_iri = min(curr_sum/n, 40)
    return to_ret_iri, x

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
parser.add_argument('--output_path', nargs='?', default='None', const='None', metavar='XXXX',
                        help='The output path to save the data')

parser.add_argument('--sample_rate', '--sr', '--sr_hz', nargs='?', type=int, default=100, const=100,
                    metavar='N', help='The sample rate of the accelerometer as the car is driving over the road profile')

parser.add_argument('--dx', '--sample_spacing', nargs='?', type=float, default=.1, const=.1,
                    metavar='X.X', help='The spacing between samples (in m) when the car drives over the profile')
parser.add_argument('--velocities', '--vel', nargs='*',
                    metavar='[V_1, V_2, ....]', help='The velocities of the cars (in m/s) to use for the simulation')

parser.add_argument('-l', '--lengths', '--profile_lengths', nargs='*',
                    metavar='[N_1, N_2, ...]', help='The lengths (in m) of the profiles to generate')
parser.add_argument('--label', '--lab',  nargs='?', default='car', const='car',
                    metavar='LABEL', help='The label of the file for identification after')




args = parser.parse_args()
#print(args.output_path)
# We want to run the simulation for each road profile, generate a csv file with the acceleration of the
# sprung mass, the road profile as input, the IRI, the velocity, and the sample spacing

class_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
num_classes = [args.A_profiles, args.B_profiles, args.C_profiles, args.D_profiles,
               args.E_profiles, args.F_profiles, args.G_profiles, args.H_profiles]
vehicle = qc.QC(m_s=args.m_s, m_u=args.m_u, c_s=args.c_s, k_s=args.c_s, k_u=args.k_u)

lengths = list(map(lambda x: int(x), args.lengths))
velocities = list(map(lambda x: float(x), args.velocities))
seed = 1
true_iris, computed_iris, true_lengths, true_velocities, road_classes = [], [], [], [], []

for l in lengths:
    for t, n in zip(class_types, num_classes):
        for x in range(0, n):
            orig_dists, orig_elevations = make_profile.make_profile_from_psd(t, 'sine', .01, l, seed)
            profile = roadprofile.RoadProfile(orig_dists, orig_elevations)
            true_iri = profile.to_iri()
            for v in velocities:
                T, yout, xout, dists, elevations = vehicle.run(profile, 100, v, args.sample_rate)
                accs = yout[:, -1]
                true_iris.append(true_iri)
                true_lengths.append(l)
                true_velocities.append(v)
                #compute_slopes(init_slope, accs, vels, deltas
                dx = dists[1] - dists[0]
                deltas = [dx]*len(accs)
                vels = [v]*len(accs)
                slopes = compute_slopes(0, accs + 9.8065, vels, deltas)
                ind_11 = np.where(dists >= 11)[0][0]
                init_v = np.sum(slopes[:ind_11]) * dx/(dists[ind_11])
                est_iri = compute_iri_from_slopes(slopes, deltas, vels, initial_value=np.array([[init_v], [0], [init_v],
                                                                                                [0]]))
                computed_iris.append(est_iri[0]*1000)
                road_classes.append(t)
            seed += 1
        print("Finished length {0} for road class {1}".format(l, t))

final_df = pd.DataFrame({'Est_IRI': computed_iris, 'Lengths': true_lengths, 'Velocities': true_velocities,
                         'True_IRI': true_iris, 'Road_Class': road_classes})
final_df.to_csv('{0}iri_{1}.csv'.format(args.output_path, args.label), index=False)







