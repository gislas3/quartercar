import numpy as np
import pandas as pd
from quartercar import qc, roadprofile
from experiments import computeisoroughness as cisr
from experiments import fitcarfrequencyresponse as fcfr
from experiments import runpsdexperiment as rpsdexp
from tests import make_profile as mp
import argparse, logging, multiprocessing, time, csv, sys, queue, pickle
from scipy.signal import welch, periodogram, lombscargle, stft
from scipy.stats import norm
from matplotlib import pyplot as plt




def generate_profiles(num_profiles, profile_length, section_length, outfile):
    dx = .05
    #total_plen = 10000
    #curr_plen = 0
    #seg_len = 500
    prof_types = ['A', 'B', 'C']
    seed_start = 300
    logging.debug("Starting generating profiles")
    st_time = time.time()
    for x in range(0, num_profiles):
        logging.debug("Starting generating profile {0}".format(x))
        ptime = time.time()
        dlist, elist = [], []
        curr_plen = 0
        while curr_plen < profile_length:
            ptype = np.random.choice(prof_types)
            distances, elevations, true_gn0 = mp.make_profile_from_psd(ptype, 'sine', dx, section_length, seed=seed_start, ret_gn0=True)
            if len(dlist) != 0:
                distances = distances + dlist[-1][-1] + dx
                elevations[0] = (elevations[1] + elist[-1][-1])/2
                distances = distances[:-1]
                elevations = elevations[:-1]
            dlist.append(distances)
            elist.append(elevations)
            curr_plen += section_length
            seed_start += 1
        distances_final = np.concatenate(dlist)
        elevations_final = np.concatenate(elist)
        np.save("{0}_{1}_{2}".format(outfile, 'distances', x), distances_final)
        np.save("{0}_{1}_{2}".format(outfile, 'elevations', x), elevations_final)
        logging.debug("Successfully saved elevations and distances, for x={0}, time was {1} seconds".format(x, time.time() - ptime))
    logging.debug('Generating {0} profiles took {1} seconds'.format(num_profiles, time.time()-st_time))
    # #profile_len = 1000
    
    #d2, e2, tn02 = mp.make_profile_from_psd('C', 'sine', dx, profile_len, seed=3, ret_gn0=True)
    
def get_window_list():
    return ['hann', 'boxcar', 'triang', 'blackman', 'hamming', 'bartlett', 'flattop', 'parzen', 'bohman', 'blackmanharris',
                                'nuttall', 'barthann', 'kaiser', 'gaussian', 'dpss', 'chebwin', 
                                'exponential', 'tukey']


def run_window_experiment(profile_list, outfile, velocity, sample_rate, add_noise, noise_sigma, 
                        make_accs, car_dict_file):
    car_list = rpsdexp.get_car_list()
    
    #next, run accelerations over all the cars - store
    car_dict = {}
    if make_accs:
        for car in car_list:
            c_start_time = time.time()
            cname, vehicle = car[0], car[1]
            logging.debug("Starting journeys for car {0}".format(cname))
            omega_s = np.sqrt(vehicle.k2)
            epsilon = 1/vehicle.mu
            omega_u = np.sqrt(vehicle.k1 * epsilon)
            xi = vehicle.c/(2*omega_s)
            car_dict[cname] = {'accelerations': [], 'true_epsilon': epsilon, 'true_omega_s': omega_s, 'true_omega_u': omega_u, 'true_xi': xi}
            for pnum, profile in enumerate(profile_list): #drive the car over all the profiles
                logging.debug("Starting driving over profile {0}".format(pnum))
                T, yout, xout, new_dists, new_els, vs = vehicle.run2(profile, [], velocity, final_sample_rate=sample_rate)
                if add_noise:
                    noise = np.random.normal(scale=np.sqrt(noise_sigma), size=len(yout[:, -1]))
                else:
                    noise = np.zeros(len(yout[:, -1]))
                car_dict[cname]['accelerations'].append(yout[:, -1] + noise) #append the accelerations
            logging.debug("Finished driving over profiles for car {0}, took {1} seconds".format(cname, time.time() - c_start_time))
        try:
            with open(car_dict_file, 'wb') as f:
                pickle.dump(car_dict, f)
        except Exception as e:
            logging.warning("Failed saving car accelerations, error was {0}. Continuing program, but car accs won't be saved :(".format(e))
    else:
        logging.debug("Reading in car trip data from file {0}".format(car_dict_file))
        try:
            with open(car_dict_file, 'rb') as f:
                car_dict = pickle.load(f)
        except Exception as e:
            logging.error("Attempted to read in car accelerations from file {0} but failed, exiting program".format(car_dict_file))
            sys.exit(1)
    
    #now, compute the PSD and fit for each window/nperseg combination
    output_dict = {'car': [], 'profile_num': [], 'window': [], 'nperseg': [], 'overlap': [], 'true_epsilon': [], 
    'true_omega_s': [], 'true_omega_u': [], 'true_xi': [], 'est_epsilon': [], 'est_omega_s': [], 
    'est_omega_u': [], 'est_xi': [], 'est_c': [], 'est_cov': []}
    npersegs = [128, 256, 512, 1024, 2048]
    overlaps = [0, 2, 4, 8]
    window_list = get_window_list()
    #kaiser: 14 (maybe dig into this more if results are promising: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.kaiser.html#scipy.signal.windows.kaiser)
    #Gaussian: Try two different sigmas, one for narrower vs. wider window
    #dpss: Needs bandwidth (going to use 5 for this)
    #chebyshev: 50 dB (not sure why honestly)
    #exponential: -(M-1) / ln(x) (x - # of samples remaining at end, let's use 20%, so M = npserseg, x = .2*nperseg)
    #tukey: Let's try .8 as well (can tweak the two above if need be)
    #for each window, overlap, and nperseg - calculate the welch psd, fit the PSD estimate, save to database
    for cname in car_dict:
        car_st_time = time.time()
        logging.debug("Starting PSD computations for car {0}".format(cname))
        car_acc_list = car_dict[cname]['accelerations']
        for pnum, acc in enumerate(car_acc_list):
            logging.debug("Starting PSD computation for profile {0}".format(pnum))
            for nps in npersegs:
                logging.debug("Starting NPS: {0}".format(nps))
                for ov in overlaps:
                    logging.debug("Starting overlap: {0}".format(ov))
                    if ov == 0:
                        overlap = 0
                    else:
                        overlap = nps//ov
                    for w in window_list: #loop through windows
                        if w == 'kaiser':
                            window = (w, 14)
                        elif w == 'gaussian':
                            window = (w, .68*nps)
                        elif w == 'dpss':
                            window = (w, 5)
                        elif w == 'chebwin':
                            window = (w, 80)
                        elif w == 'exponential':
                            window = (w, None, -(nps-1)/np.log(.2))
                        elif w == 'tukey':
                            window = (w, .8)
                        else:
                            window = w
                            
                        f_acc, acc_psd = welch(acc, fs=sample_rate, window=window, nperseg=nps, noverlap=overlap)
                        #plt.loglog(f_acc, acc_psd)
                        #plt.show()
                        try:
                            p_est, cov_est = fcfr.fit(f_acc, acc_psd, fmin=5*(sample_rate)/nps, fmax=25, params0=[6, 7, 50, .1, 2000*1e-8], bounds=([1, .001, .1, 0, 0], [50, 100, 100, 30, 1000000]), sigma=np.array(noise_sigma*np.ones(len(acc_psd))))
                        except Exception as e:
                            logging.warning("curve fitting failed for Window size {0}, Window {5}, ov {1}, car {2}, profile {3}, exception was {4}".format(nps, overlap/nps, cname, pnum, e, w))
                            p_est = [np.nan, np.nan, np.nan, np.nan, np.nan]
                            cov_est = np.nan
                        output_dict['car'].append(cname)
                        output_dict['profile_num'].append(pnum)
                        output_dict['window'].append("{0}".format(window))
                        output_dict['nperseg'].append(nps)
                        output_dict['overlap'].append(overlap/nps)
                        output_dict['true_epsilon'].append(car_dict[cname]['true_epsilon'])
                        output_dict['true_omega_s'].append(car_dict[cname]['true_omega_s'])
                        output_dict['true_omega_u'].append(car_dict[cname]['true_omega_u'])
                        output_dict['true_xi'].append(car_dict[cname]['true_xi'])
                        output_dict['est_epsilon'].append(p_est[0])
                        output_dict['est_omega_s'].append(p_est[1])
                        output_dict['est_omega_u'].append(p_est[2])
                        output_dict['est_xi'].append(p_est[3])
                        output_dict['est_c'].append(p_est[4])
                        output_dict['est_cov'].append(cov_est)
        logging.debug("Finished Car {0}, took {1} seconds".format(cname, time.time() - car_st_time))
    try:
        with open(outfile, 'wb') as f:
            pickle.dump(output_dict, f)
    except Exception as e:
        logging.error("failed saving the output dictionary, printing to standard output in hopes that it will be useful:\n {0}".format(output_dict))
        
        


    
    





#if __name__ == '__main__':
parser = argparse.ArgumentParser(
        description="""Program for running/saving the results of finding the optimal window size/overlap for the random profiles""")

parser.add_argument('--gen_profs', '--generate_profiles', '--make_profiles', nargs='?', type=int,
                        default=0,
                        const=0,
                        metavar='X',
                        help='Flag denoting whether or not to generate the road profiles')
parser.add_argument('--num_profs', '--num_profiles', '--n_profiles', '--n_profs', nargs='?', type=int,
                        default=15,
                        const=15,
                        metavar='N',
                        help='The number of profiles to generate')
parser.add_argument('--prof_len', '--profile_length', '--p_length', nargs='?', type=int,
                        default=10000,
                        const=10000,
                        metavar='N',
                        help='The length of the profiles to generate (only used if gen_profs != 0)')
parser.add_argument('--sec_len', '--section_length', '--s_length', nargs='?', type=int,
                        default=500,
                        const=500,
                        metavar='N',
                        help='The length of the subsection of the profiles to generate (only used if gen_profs != 0)')
parser.add_argument('--vel', '--velocity', '--speed', nargs='?', type=float, 
                        default=15,
                        const=15,
                        metavar='V (m/s)',
                        help='The velocity at which to run the simulation at')
parser.add_argument('--outfile', '--out_file', '--title', nargs='?',
                        default='/Users/gregoryislas/Documents/Mobilized/data_dump/PSD_Exp/unknown.csv',
                        const='/Users/gregoryislas/Documents/Mobilized/data_dump/PSD_Exp/unknown.csv',
                        metavar='file_path/experiment_title',
                        help='The name of the output file, supposed to be used as descriptor for parameters of experiment ')
parser.add_argument('--prof_direc', '--profile_directory', '--profile_direc', nargs='?',
                    default='test_profiles',
                    const='test_profiles',
                    metavar='[PATH_TO_PROFILES]',
                    help='The directory where the profiles for the experiment have been saved')
parser.add_argument('--sr', '--sample_rate', '--sample_rate_hz', '--hertz', '--hz', nargs='?',
                        type=int,
                        default=100,
                        const=100,
                        metavar='X',
                        help='The sample rate (in the time domain) at which to sample the acceleration series. ' +
                             'Maximum of 1000 Hz')
parser.add_argument('--log_level', '--log_lev', '--loglevel', '--log',  nargs='?',
                        default='DEBUG',
                        const='DEBUG',
                        metavar='DEBUG INFO WARNING ERROR',
                        help='The level of logging to use')
parser.add_argument('--noise', '--add_noise',  nargs='?', type=int,
                        default=1,
                        const=1,
                        metavar='T/F (1/0)',
                        help='Flag denoting whether or not to add gaussian noise to the acceleration series (default: True)')
parser.add_argument('--noise_sigma', '--noise_sig',  nargs='?', type=float,
                        default=1e-2,
                        const=1e-2,
                        metavar='sigma**2',
                        help='The standard deviation of the gaussian noise to add to the acceleration series (default: 1e-2)')
parser.add_argument('--make_accs', '--drive',  nargs='?', type=int,
                        default=1,
                        const=1,
                        metavar='T/F (1/0)',
                        help='Flag denoting whether to create the acc')
parser.add_argument('--acc_file', '--acc_files', '--car_accs',  nargs='?',
                        default='test_accs.pkl',
                        const='test_accs.pkl',
                        metavar='[FILE_NAME]',
                        help='File name to read in the precomputed car accelerations (only used of make_accs == 1)')
args = parser.parse_args()

level = args.log_level
if level == 'DEBUG':
    loglevel = logging.DEBUG
elif level == 'INFO':
    loglevel = logging.INFO
elif level == 'WARNING':
    loglevel = logging.WARNING
elif level == 'ERROR':
    loglevel == logging.ERROR
else:
    loglevel = logging.info
logging.basicConfig(format='%(asctime)s: %(levelname)s:%(message)s', level=loglevel)
if args.gen_profs:
    generate_profiles(args.num_profs, args.prof_len, args.sec_len, args.outfile)
else:
    try:
        profile_list = []
        profile_direc = args.prof_direc
        num_profiles = args.num_profs
        for x in range(0, num_profiles):
            ds_temp, es_temp = np.load("{0}_{1}_{2}.npy".format(profile_direc, 'distances', x)), np.load("{0}_{1}_{2}.npy".format(profile_direc, 'elevations', x))
            prof_temp = roadprofile.RoadProfile(distances=ds_temp, elevations=es_temp)
            profile_list.append(prof_temp)
        run_window_experiment(profile_list, args.outfile, args.vel, args.sr, args.noise, args.noise_sigma, 
                        args.make_accs, args.acc_file)
    except Exception as e:
        logging.error("Attempted to read in profiles from directory {0} but failed, exiting program with status 1. Exception was {1}".format(profile_direc, e))
        sys.exit(1)