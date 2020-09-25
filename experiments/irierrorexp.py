import numpy as np
import pandas as pd
from quartercar import qc, roadprofile
from experiments import computeisoroughness as cisr
#from experiments import fitcarfrequencyresponse as fcfr
from experiments import runpsdexperiment as rpsdexp
from tests import make_profile as mp
import argparse, logging, time, csv, sys, queue, pickle
#from scipy.signal import welch, periodogram, lombscargle, stft
#from scipy.stats import norm
from matplotlib import pyplot as plt

def get_car_list():
    
    #car1 = qc.QC(epsilon=240/36, omega_s=np.sqrt(16000/240), omega_u=np.sqrt(160000/36), xi=980/(2*np.sqrt(16000*240)))
    car1 = qc.QC(m_s=240, m_u=36, k_s=16000, c_s=980, k_u=160000)
    car2 = qc.QC(m_s=250, m_u=40, k_s=28000, c_s=2000, k_u=125000)
    car3 = qc.QC(m_s=208, m_u=28, k_s=18709, c_s=1300, k_u=127200)
    car4 = qc.QC(m_s=300, m_u=50, k_s=18000, c_s=1200, k_u=180000)
    car5 = qc.QC(m_s=243, m_u=40, k_s=14671, c_s=370, k_u=124660)
    car6 = qc.QC(m_s=257, m_u=31, k_s=13100, c_s=400, k_u=126000)
    car7 = qc.QC(m_s=290, m_u=59, k_s=16812, c_s=1000, k_u=190000)
    suv1 = qc.QC(m_s=650, m_u=55, k_s=27500, c_s=3000, k_u=237000)
    suv2 = qc.QC(m_s=737.5, m_u=62.5, k_s=26750, c_s=3500, k_u=290000)
    bus = qc.QC(m_s=4500, m_u=500, k_s=300000, c_s=20000, k_u=1600000)

    return [('car1', car1), ('car2', car2), ('car3', car3), ('car4', car4), 
    ('car5', car5), ('car6', car6), ('car7', car7), ('suv1', suv1), ('suv2', suv2), ('bus', bus)]

def run_iri_error_exp(profile_list, outfile, velocity, sample_rate, noise, noise_sigma, e_param, seed):
    e_dict = {'profile_num': [], 'profile_type': [], 'true_iri': [], 'car': [], 
    'error_pct': [], 'error_param_name': [], 'error_param': [], 'true_param': [], 'estimated_iri': []}
    e_pcts = [0, .1, .25, .5, .75, 1]
    np.random.seed(seed)
    beg_time = time.time()
    for e_pct in e_pcts:
        for car_tup in get_car_list():
            st_time = time.time()
            cname, vehicle = car_tup[0], car_tup[1]
            logging.debug("Starting error param {0}, pct {1}, car {2}".format(e_param, e_pct, cname))
            
            omega_s = np.sqrt(vehicle.k2)
            epsilon = 1/vehicle.mu
            omega_u = np.sqrt(vehicle.k1 * epsilon)
            xi = vehicle.c/(2*omega_s)
            v_params = {'epsilon': epsilon, 'omega_s': omega_s, 'omega_u': omega_u, 'xi': xi}
            ev_params = {'epsilon': epsilon, 'omega_s': omega_s, 'omega_u': omega_u, 'xi': xi}
            if e_param == 'epsilon':
                ekeys = ['epsilon']
            elif e_param == 'omega_s':
                ekeys = ['omega_s']
            elif e_param == 'omega_u':
                ekeys = ['omega_u']
            elif e_param == 'xi':
                ekeys = ['xi']
            else:
                ekeys = ['epsilon', 'omega_s', 'omega_u', 'xi']
            for k in ekeys: 
                new_param = (1 + np.random.choice([1, -1])*e_pct)*ev_params[k]
                if new_param <= 0:
                    new_param = (1 + e_pct)*ev_params[k] # in case less than 0
                ev_params[k] = new_param
            error_car = qc.QC(**ev_params)
            for pnum, prof_tup in enumerate(profile_list):
                ptype, profile = prof_tup[0], prof_tup[1]
                true_iri = profile.to_iri()
                T, yout, xout, new_dists, new_els, vs = vehicle.run2(profile, [], velocity, final_sample_rate=sample_rate)
                input_times, input_accs, elevations = error_car.inverse_time(T, yout[:, -1], sample_rate, interp_dt=.01, interp_type='linear', 
                    Wn_integrate=None, Wn_elevations=None, f_order=4, space_evenly=False, velocity=0)
                new_ds = velocity * input_times
                profile_est = roadprofile.RoadProfile(new_ds, elevations)
                iri_est = profile_est.to_iri()
                e_dict['profile_num'].append(pnum) 
                e_dict['profile_type'].append(ptype)
                e_dict['true_iri'].append(true_iri)
                e_dict['car'].append(cname)
                e_dict['error_pct'].append(e_pct)
                e_dict['error_param_name'].append(e_param)
                eps, tps = [], []
                for k in ekeys:
                    eps.append(ev_params[k])
                    tps.append(v_params[k])
                e_dict['error_param'].append(eps) 
                e_dict['true_param'].append(tps)
                e_dict['estimated_iri'].append(iri_est)
            logging.debug("Took {0} seconds to run over profiles for car {1}".format(time.time() - st_time, cname))
    try:
        with open(outfile, 'wb') as f:
            pickle.dump(e_dict, f)
    except Exception as e:
        logging.error("failed saving the output dictionary, printing to standard output in hopes that it will be useful:\n {0}".format(e_dict))
    logging.debug("Took {0} seconds for all cars to run over profiles".format(time.time() - beg_time))
                 


            






def generate_profiles(num_profiles, profile_length, outfile):
    dx = .05
    #total_plen = 10000
    #curr_plen = 0
    #seg_len = 500
    prof_types = ['A', 'B', 'C']
    prof_nums = [num_profiles//3 + num_profiles%3, num_profiles//3, num_profiles//3]
    seed_start = 1
    logging.debug("Starting generating profiles")
    st_time = time.time()
    for ptype, pnum in zip(prof_types, prof_nums):
        for x in range(0, pnum):
            ptime = time.time()
            logging.debug("Starting generating profile {0} class {1}".format(x, ptype))
            distances, elevations, true_gn0 = mp.make_profile_from_psd(ptype, 'sine', dx, profile_length, seed=seed_start, ret_gn0=True)
            seed_start += 1
            np.save("{0}_{1}_{2}_{3}".format(outfile, 'distances', ptype, x), distances)
            np.save("{0}_{1}_{2}_{3}".format(outfile, 'elevations', ptype, x), elevations)
            logging.debug("Successfully saved elevations and distances, for x={0}, time was {1} seconds".format(x, time.time() - ptime))
    logging.debug('Generating {0} profiles took {1} seconds'.format(num_profiles, time.time()-st_time))



parser = argparse.ArgumentParser(
        description="""Program for running/saving the results of computing the iri error for wrong parameters in QC model""")

parser.add_argument('--gen_profs', '--generate_profiles', '--make_profiles', nargs='?', type=int,
                        default=0,
                        const=0,
                        metavar='X',
                        help='Flag denoting whether or not to generate the road profiles')
parser.add_argument('--num_profs', '--num_profiles', '--n_profiles', '--n_profs', nargs='?', type=int,
                        default=30,
                        const=30,
                        metavar='N',
                        help='The number of profiles to generate')
parser.add_argument('--prof_len', '--profile_length', '--p_length', nargs='?', type=int,
                        default=500,
                        const=500,
                        metavar='N',
                        help='The length of the profiles to generate (only used if gen_profs != 0)')

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

parser.add_argument('--e_param', '--error_param', nargs='?',
                        default='epsilon',
                        const='epsilon',
                        metavar='[PARAM_NAME]',
                        help='The parameter under which to consider the error in IRI based on parameter error. Default is epsilon, options are: epsilon, omega_s, omega_u, xi, all')

parser.add_argument('--seed', nargs='?', type=int,
                        default=69,
                        const=69,
                        metavar='[RANDOM_SEED]',
                        help='The random seed to use for the parameter errors')


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
    generate_profiles(args.num_profs, args.prof_len, args.outfile)
else:
    try:
        profile_list = []
        profile_direc = args.prof_direc
        num_profiles = args.num_profs
        ptypes = ['A', 'B', 'C']
        prof_nums = [num_profiles//3 + num_profiles%3, num_profiles//3, num_profiles//3]
        for ptype, pnum in zip(ptypes, prof_nums):
            for x in range(0, pnum):
                ds_temp, es_temp = np.load("{0}_{1}_{2}_{3}.npy".format(profile_direc, 'distances', ptype, x)), np.load("{0}_{1}_{2}_{3}.npy".format(profile_direc, 'elevations', ptype, x))
                prof_temp = roadprofile.RoadProfile(distances=ds_temp, elevations=es_temp)
                profile_list.append((ptype, prof_temp))
        run_iri_error_exp(profile_list, args.outfile, args.vel, args.sr, args.noise, args.noise_sigma, 
                        args.e_param, args.seed)
    except Exception as e:
        logging.error("failed reading in profiles, exiting program, exception was {0}".format(e))