import argparse, logging, time, pickle, gzip, sys
import numpy as np
from quartercar import qc, roadprofile, cars
from multiprocessing import Pool, Lock


# This script will simulate the accelerations from all of the cars defined in the cars.py file for each road profile,
# at the requested velocities


# Architecture of program - we will have multiple processes creating the acceleration data, and we 
# will have multiple processes taking in the acceleration data, and then estimating the parameters, all that stuff
# 
#def generate_accs(road_types, road_lengths, num_roads, velocities, output_directory):
def generate_accs(road_type, road_length, road_number, profile, velocities, gn0, output_directory, log_lock):
    """
    generate_accs() will simulate several cars driving over a road profile at different velocities, and then save the output to a file.
    :param road_type: The class of the road profile (should be one of A, B, C, D, E, F, G or H - will likely always be A, B, or C). Only used for saving the output file.
    :param road_length: The length of the road profile, only used for saving the output file
    :param road_number: The number of the road profile, only used for saving the output file
    :param profile: A `RoadProfile` instance that defines the road that the cars will be driving over
    :param velocities: A list of velocities (in units m/s) that define the speeds over which the simulations should take place.
    :param gn0: The "roughness parameter" of the road profile.
    :param output_directory: The directory at which to save the output of the simulations.
    :param log_lock: If using multiprocessing, a handy lock so that two processes don't try to access the logging file at the same time
    
    """
    data_dict = {'gn0': [], 'car': [], 'velocity': [], 'sprung_accelerations': [], 'sprung_velocities': [], 'sprung_displacements': [], 
    'unsprung_velocities': [], 'unsprung_displacements': [], 'input_elevations': []} #the format of which to store the data - each file will be a compressed pickle file
    #data_dict = {'gn0': [], 'car': [], 'velocity': [], 'sprung_accelerations': [], 'input_elevations': []} 
    #for rt in road_types:
    #    for rl in road_lengths:
    #        st_time = time.time()
    #        for n in range(0, num_roads):
    #            dists = np.load('{0}Profile_{1}_{2}_{3}_distances.npy'.format(profile_directory, rt, rl, n))
    #            elevations = np.load('{0}Profile_{1}_{2}_{3}_elevations.npy'.format(profile_directory, rt, rl, n))
    #            gn0 = np.load('{0}Profile_{1}_{2}_{3}_gn0.npy'.format(profile_directory, rt, rl, n))
    #            curr_profile = roadprofile.RoadProfile(distances=dists, elevations=elevations)
    #st_time = time.time() #Just in case want to know how long these simulations are taking
    for car_tup in cars.get_car_list(): 
        cname = car_tup[0]
        car = car_tup[1]
        for vel in velocities:
            #print("Onwards we go")
            #logging.info("Onwards we go")
            T, yout, xout, new_dists, new_els, _ = car.run2(profile, [], vel, final_sample_rate=1000) #always sample at 1000 Hz, the max rate
            sprung_accs = yout[:, -1] #get the sprung accelerations
                        #xout[:, 1], xout[:, 0], xout[:, 3], xout[:, 2]
                        #true_vels, true_disps, us_vels, us_disps
            sprung_vels = xout[:, 1] #sprung velocities
            sprung_disps = xout[:, 0] #you get the idea
            unsprung_vels = xout[:, 3]
            unsprung_disps = xout[:, 2]
                        #data_dict['road_type'].append(rt)
                        #data_dict['length'].append(rl)
                        #data_dict['road_number'].append(n)
            #append all the data
            data_dict['gn0'].append(gn0)
            data_dict['car'].append(cname)
            data_dict['velocity'].append(vel)
            data_dict['input_times'].append(T) #no use appending the times, they can easily be computed from the velocity, sample rate, and profile length
            data_dict['sprung_accelerations'].append(sprung_accs)
            data_dict['sprung_velocities'].append(sprung_vels)
            data_dict['sprung_displacements'].append(sprung_disps)
            data_dict['unsprung_velocities'].append(unsprung_vels)
            data_dict['unsprung_displacements'].append(unsprung_disps)
            data_dict['input_distances'].append(new_dists)
            data_dict['input_elevations'].append(new_els)
                        #)
            #logging.info("Took {0} seconds for one round of length {1}, class {2}".format(time.time() - st_time, rl, rt))
    #save the file
    # with gzip.open("{0}/{1}_{2}_{3}_accs.pickle.gz".format(output_directory, road_type, road_length, road_number), 'wb') as f:
    #     pickle.dump(data_dict, f)
    
    # #write the log messages
    # if log_lock is not None:
    #     try:
    #         log_lock.acquire()
    #         logging.info("Took {0} seconds for {1},{2},{3}".format(time.time() - st_time, road_type, road_length, road_number))
        
    #     finally:
    #         log_lock.release()
    # else:
    #     logging.info("Took {0} seconds for {1},{2},{3}".format(time.time() - st_time, road_type, road_length, road_number))
    return data_dict

# def generate_accs_star(args):
#     """
#     A helper method that just calls generate_accs, since the default behavior of python's imap and imap_undordered methods is to pass in 
#     the data as one argument. More info here: https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.imap
#     :param args: The arguments (should be in a dictionary) to pass into generate_accs()
#     """
#     #logging.info("Generating accs")
#     return generate_accs(**args)
#     #logging.info("Generated accs")

def acc_input_data_generator(profile_directory, road_types, road_lengths, num_roads, velocities, output_directory, sample_rates, acc_noise, 
    F_known, Q_known, fu_mu_known, n_param_inits, order=1):
    """
    This method defines a generator that will be used to feed the input data into generate_accs.
    :param profile_directory: The directory from which to read the input road profiles, which should be saved in a specified format as seen below
    :param road_types: A list representing the road types of the profiles, should be one of A, B, C, D, E, F, G, or H
    :param road_lengths: A list representing the road lengths (in units m) of the profiles to read in.
    :param num_roads: The number of total roads *per class*. 
    :param velocities: The list of velocities to drive over said profile at (in units m/s)
    :param output_directory: The output_directory at which to save the generated acceleration data
    :param log_lock: A `multiprocessing.Lock` object to pass to generate_accs() so the log writing to files is process/thread safe
    """
    #total_to_process = len(road_types) * len(road_lengths) * 
    for rt in road_types:
        for rl in road_lengths:
            for n in range(0, num_roads):
                dists = np.load('{0}Profile_{1}_{2}_{3}_distances.npy'.format(profile_directory, rt, rl, n))
                elevations = np.load('{0}Profile_{1}_{2}_{3}_elevations.npy'.format(profile_directory, rt, rl, n))
                gn0 = np.load('{0}Profile_{1}_{2}_{3}_gn0.npy'.format(profile_directory, rt, rl, n))
                curr_profile = roadprofile.RoadProfile(distances=dists, elevations=elevations)
                yield {'road_type': rt, 'road_length': rl, 'road_number': n, 'profile': curr_profile,
                'velocities': velocities, 'gn0': gn0, 'output_directory': output_directory, 'acc_noise': acc_noise,
                'F_known': F_known, 'Q_known': Q_known, 'fu_mu_known': fu_mu_known, 'sample_rates': sample_rates, 'n_param_inits': n_param_inits,
                'order': order}


    #generate_accs(args.profile_directory, args.road_types, args.road_lengths, args.num_roads, args.velocities, args.output_directory[0])
    #logging.info("Total program run time: {0} seconds".format(time.time() - st_time))

#parse_args_and_run()   
    
