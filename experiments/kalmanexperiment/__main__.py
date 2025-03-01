import argparse, logging
from multiprocessing import Pool
from experiments.kalmanexperiment import kalmangenerateaccs, emexp





def parse_args_and_run():
    """
    parse_args_and_run() takes in the input from the command line, and handles starting this program/the potential multiprocessing. See the 
    description in the ArgumentParser for more info.
    """
    parser = argparse.ArgumentParser(
        description="""Program for estimating vehicle parameters and road profiles using the Expectation-Maximization algorithm with a 
        kalman filter state-space setup. Currently, can only test with constant sample rates and constant speeds. Future improvements will allow for
        non-Gaussian noise, constant force inputs to simulate braking/accelerating, orientation estimation with ramp inputs, non-constant speeds/sampling rates,
        and possibly data generated from a half-car or full-car model.""")

    parser.add_argument('--profile_directory', '--prof_direc', '--prof_directory', '--profile_direc', nargs='?',
                        default='/Users/gregoryislas/Documents/Mobilized/data_dump/Road_Profiles/',
                        const='/Users/gregoryislas/Documents/Mobilized/data_dump/Road_Profiles',
                        metavar='PATH_TO_PROFILES',
                        help='The directory where the profiles reside over which to drive the car over')
    
    parser.add_argument('--road_types', '--rts', '--road_class', '--road_classes',  nargs='*',
                        metavar='[A, B, C, .... ]',
                        help='The types of road profiles')
    
    parser.add_argument('--road_lengths', '--road_lens', '--rls', '--lengths',  nargs='*',
                        metavar='[len1, len2, ..., lenn]',
                        help='The lengths of the road profiles')

    parser.add_argument('--log_level', '--log_lev', '--loglevel', '--log',  nargs='?',
                        default='INFO',
                        const='INFO',
                        metavar='DEBUG INFO WARNING ERROR',
                        help='The level of logging to use')

    parser.add_argument('--log_file', '--log_f', '--logfile', '--lfile',  nargs='?',
                        default='/Users/gregoryislas/Documents/Mobilized',
                        const='/Users/gregoryislas/Documents/Mobilized',
                        metavar='PATH_TO_LOG_FILE',
                        help='Where to store the output from the logging')

    parser.add_argument('--num_roads', '--n_roads', '--total_roads', '--roads_per_class', nargs='?', type=int,
                        default=30,
                        const=30,
                        metavar='N',
                        help='The numer of profiles (per class)')

    
    parser.add_argument('--velocities', '--vels', nargs='*',
                        metavar='[vel1, vel2, ..., veln]',
                        help='The velocities at which to drive the cars over the profile')

    parser.add_argument('--output_directory', '--out_direc', '--output_directory', '--out_direc', nargs='?',
                        default='~/',
                        const='~/',
                        metavar='PATH_TO_OUTPUT',
                        help='The directory where to save the acceleration series generated by the cars')

    parser.add_argument('--processes', '--num_processes', '--n_processes', 
    type=int, nargs='?',  default=-1,
    const=-1, metavar='NUM_PROCESSES', 
    help="The number of parallel processes to use. If want to use number of CPUS, type in number less than 0. If don't want multiprocessing, type 0, Default: -1")

    parser.add_argument('--chunk_size', '--c_size', '--CHUNK_SIZE', nargs='?', type=int,
    default=64, const=64, metavar='CHUNK_SIZE', help='The chunk size to use for the input to the process pool, default: 64')
    
    
    parser.add_argument('--sample_rates', '--srs', '--sample_rates_hz', '--sr_hz',  nargs='*',
                        metavar='[SR_1, SR_2, ..., SR_N]',
                        help='The sampling rates at which to downsample the acceleration series')
    
    parser.add_argument('--Q_known', '--q_known', type=int, nargs='?',
    default=0,
    const=0,
    metavar='0/1',
    help='Flag indicating whether Q is supposed to be known or unknown. Default: 0 (unknown)')

    parser.add_argument('--F_known', '--F_known', type=int, nargs='?',
    default=0,
    const=0,
    metavar='0/1',
    help='Flag indicating whether F is supposed to be known or unknown. Default: 0 (unknown)')

    parser.add_argument('--fu_mu_known', '--Fu_Mu_known', type=int, nargs='?',
    default=0,
    const=0,
    metavar='0/1',
    help='Flag indicating whether Fu/mu is supposed to be known or unknown. Default: 0 (unknown). Ignored if --F_known set to 1')

    parser.add_argument('--vel_weights', '--velocity_weights', '--vws', nargs='*', 
    metavar='[Weight_1, Weight_2, ...., Weight_N]',
    help='The weights to use for randomly selecting the velocities. Needs to be the same length as the velocities if using simulation method 2')

    parser.add_argument('--rl_weights', '--road_length_weights', nargs='*',
                        metavar='[Length_Weight1, Length_Weight2, ..., Length_WeightN]',
                        help='The weights at which to use for choosing the road length')

    parser.add_argument('--acc_noise', '--acc_noise_level', '--meas_noise', nargs='*', 
    metavar='[Sigma_1, Sigma_2, ..., Sigma_N]',
    help='The noise levels to use for the accelerometer noise')

    parser.add_argument('--n_param_inits', '--n_param_init', '--param_inits', type=int, nargs='?',
    default=20,
    const=20,
    help='The number of random parameter initializations to use')

    parser.add_argument('--order', '--F_order',type=int, nargs='?',
    default=1,
    const=1,
    help="The order of the taylor series expansion to use for the discretization of the state space. Default: 1. Use 'inf' to use a numerical matrix exponential algorithm")

    parser.add_argument('--num_simulations', '--n_sims', '--simulations', '--num_sims', nargs='?', type=int,
    default=100,
    const=100,
    metavar='N_SIMULATIONS',
    help='The number of simulations to perform for this experiment')

    parser.add_argument('--seed', '--start_seed', type=int, nargs='?',
    default=1,
    const=1,
    metavar='RANDOM_SEED', 
    help='The seed to use for the random parameter initializations')

    parser.add_argument('--exp_type', '--experiment_type', nargs='?', type=int, 
    default=1,
    const=1,
    metavar='EXP_TYPE',
    help='The type of experiment to run. Type 1 for generating fixed values, type 2 for randomly selecting parameters for each experiment')
    





    args = parser.parse_args()
    
    
    #sys.exit(0)
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
        loglevel = logging.INFO
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s:%(message)s', filename=args.log_file, level=loglevel)
    logging.info(args)
    logging.debug("We are in DEBUG mode!")
    args.road_lengths = list(map(lambda x: int(x), args.road_lengths))
    args.velocities = list(map(lambda x: int(x), args.velocities))
    args.sample_rates = list(map(lambda x: int(x), args.sample_rates))
    args.acc_noise = list(map(lambda x: float(x), args.acc_noise))
    if len(args.vel_weights) > 0 and args.vel_weights[0] is not None:
        args.vel_weights = list(map(lambda x: float(x), args.vel_weights))
    else:
        args.vel_weights = len(args.velocities) * [1/len(args.velocities)]
    if len(args.rl_weights) > 0 and args.rl_weights[0] is not None:
        args.rl_weights = list(map(lambda x: float(x), args.rl_weights))
    else:
        args.rl_weights = len(args.road_lengths) * [1/len(args.road_lengths)]
    # Create the data generator that will be used to load in the road profiles, and give the input to the program which will simulate a car trip, 
    # then estimate the road profile and vehicle parameters 
    if args.exp_type == 1:
        acc_data_generator = kalmangenerateaccs.acc_input_data_generator(args.profile_directory, args.road_types, args.road_lengths,
        args.num_roads, args.velocities, args.output_directory, args.sample_rates, args.acc_noise, args.F_known, args.Q_known, args.fu_mu_known, args.n_param_inits)
    else: # If you don't type 1, you're getting the second option
        acc_data_generator = kalmangenerateaccs.acc_random_data_generator(args.profile_directory, args.road_types, args.road_lengths, args.rl_weights,
        args.num_roads, args.velocities, args.vel_weights, args.output_directory, args.sample_rates, args.acc_noise, args.F_known, args.Q_known, args.fu_mu_known, args.n_param_inits,
        args.num_simulations) #TODO: Maybe make an option for including weights for the simulation; if this gets big enough, might want to make a simulation class...
    
    #Decide whether to use multiprocessing or not
    if args.processes != 0: 
        if args.processes > 0:
            processes = args.processes
        else:
            processes = None
        with Pool(processes) as pool:
            for tup in pool.imap_unordered(emexp.estimate_star, acc_data_generator, args.chunk_size):
                logging.info("Took {0} seconds to run Class: {1}, Length: {2}, #: {3}, n_sim: {4} ".format(tup[0], tup[1], tup[2], tup[3], tup[4]))
                #pass
    else: #This will probably be very slow, but good for testing
        for val in acc_data_generator:
            tup = emexp.estimate_star(val)
            logging.info("Took {0} seconds to run Class: {1}, Length: {2}, #: {3} ".format(tup[0], tup[1], tup[2], tup[3]))
        #pass #TODO: Make work without multiprocesses

    #start_processing_accs(args.input_file[0], args.output_file[0], args.profile_directory, args.num_roads, args.sample_rates, args.Q_known, args.F_known, 
    #args.fu_mu_known, args.acc_noise, args.n_param_inits, args.order, args.seed)
    #create the logger to be used

    #st_time = time.time()
    #Make sure these are ints, the python default is to make them strings
    
    
    #log_lock = Lock()

    #Create the data generator
    


    

    
parse_args_and_run()
#kalmangenerateaccs.parse_args_and_run()
#emexp.parse_args_and_run()