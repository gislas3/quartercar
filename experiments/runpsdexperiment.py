import numpy as np
import pandas as pd
from quartercar import qc, roadprofile
from experiments import computeisoroughness as cisr
from tests import make_profile as mp
import argparse, logging, multiprocessing, time, csv, sys, queue
from scipy.signal import welch, periodogram, lombscargle, stft
from scipy.stats import norm
from matplotlib import pyplot as plt





#TODO: Convert power spectral density to proper spatial frequencies based on transfer function

def rmse_frequency_bins(f_true, psd_true, f_est, psd_est):
    """
    Computes the errors at each frequency bin for the power spectral density estimtate derived from using
    the transfer function
    :param f_true: The true spatial frequencies (in units cycles/meter) for the road profile
    :param psd_true: The true power spectral density for the road profile
    :param f_est: The estimated spatial frequencies derived from the time series of accelerometer measurements
    :param psd_est: The estimated spatial power spectral density derived from the time series of accelerometer measurements
    :return: frequencies: the frequency bins at which the RMSE was computed
            errors: The difference between the true power spectral density and the estimated one at each frequency bin
    """
    #Will compute rmse for PSD recovery using frequency bins/PSD from road vs. recovered from acceleration history
    #will use the bins for the higher resolution (higher resolution = shorter bin)
    freq_res1 = f_true[1] - f_true[0]
    freq_res2 = f_est[1] - f_est[0]
    if freq_res1 > freq_res2: #lower resolution for original psd
        ratio = freq_res2/freq_res1
        psd_to_avg = psd_est
        psd_ref = psd_true
        final_res = freq_res1
        #in this case, the reference psd is the true one - so will already make the averaged one the same length, with added zeros, so don't need to do anything
    else:
        ratio = freq_res1/freq_res2
        psd_to_avg = psd_true
        psd_ref = psd_est
        final_res = freq_res2
        #in this case, the psd to average is the true psd - so we want to compare against all its frequencies, rather than
        #just the frequencies of the estimated psd (and concat higher frequencies if estimated has higher)
        len_pref = int(np.max(f_true)/final_res) + 1
        #print(len_pref)
        if len(psd_ref) < len_pref:
            psd_ref = np.concatenate((psd_ref, np.zeros(len_pref - len(psd_ref))))
        elif len(psd_ref) > len_pref:
            psd_ref = psd_ref[:len_pref]




    #assume that both frequency lists start at 0
    weights = []
    start = 0
    prev_overlap = ratio

    psd_compare = np.zeros(len(psd_ref))

    psd_compare[0] = psd_to_avg[0]
    #fmax = len(psd_compare) * final_res
    #f_curr = final_res
    idx = 1
    c_idx = 1
    #n_to_use = int(np.ceil(1 / ratio))  # how many psd values to use for the average
    while idx < len(psd_to_avg) and c_idx < len(psd_compare):

        st_overlap = (ratio - prev_overlap)%ratio
        w_overlap = st_overlap/final_res
        n_equal_weights = int((final_res - st_overlap)//ratio)
        end_overlap = final_res - (n_equal_weights * ratio + st_overlap)
        w_end = end_overlap/final_res
        weights = int(st_overlap > 0)*[w_overlap] + n_equal_weights*[ratio] + int(end_overlap > 0)*[w_end]
        to_avg = psd_to_avg[idx:idx + len(weights)]
        psd_compare[c_idx] = np.average(to_avg, weights=weights[:len(to_avg)])
        c_idx += 1
        idx += n_equal_weights + int(st_overlap > 0)
        #print(int(st_overlap > 0))
        #print(end_overlap)

        #print(n_equal_weights + int(st_overlap > 0))
        #f_curr = f_curr + final_res
        prev_overlap = end_overlap




    #idx = 0
    #c_idx = 0
    #while idx < len(psd_to_avg) and c_idx < len(psd_compare): #stop when reach end in case the two don't cover the same frequency ranges
    #    n_equal_weights = int((1/ratio)//1) #1/ratio = number of bins to use, full bins is 1/ratio//1
    #    wts = n_equal_weights*[ratio] + [((1/ratio) % 1) * ratio]*(n_to_use - n_equal_weights) #use ratio for full bins,
    #    to_avg = psd_to_avg[idx:idx+n_to_use]
    ##    psd_compare[c_idx] = np.average(to_avg, weights=wts[:len(to_avg)]) #cut off in case of end
    #    idx += n_equal_weights
    #    c_idx += 1
    #psd_ref = psd_ref[:len(psd_compare)] #question - count frequencies not captured as error
    errors = psd_ref - psd_compare
    fmax = len(psd_compare) * final_res
    if sum(np.isnan(errors) != 0):
        print("Psd ref is {0}".format(psd_ref))
        print("Psd compare is {0}".format(psd_compare))
    return np.arange(0, fmax, final_res), errors
    #TODO: Write frequency averaging for comparing the errors of the PSD estimate




def average_psd_freq_bins(frequencies, psd):
#averages the psd over different spatial frequency bins
    pass

def average_stft_psd(car, orig_times, st_times, frequencies, stf, velocities, f_min):
    #computes the spatial psd based on averaging spatial frequencies over windows
    #we won't use the averages for the low frequencies for windows that weren't able to capture the minimum frequency

    #Need to match velocities to the stft times - since we cannot compute the exact FFT for each point, we are going
    #to use the average velocity over the time period (will likely need to do some checking to make sure it doens't vary too much)



    return None, None


def compute_psd_varying_veloc(car, times, accelerations, velocities, sample_rate_hz, f_min, window_function, overlap):
    #Assumes that frequencies are in cycles/second, and can be converted to cycles per meter by dividing by velocity
    #Need to compute the nperseg based on the minimum velocity, minimum frequency needed, and sampling rate
    len_series = len(accelerations)
    #Usually, f_min will be .01 cycles per meter (so a wavelength of 100 m) - obviously this will need to be tweaked based on the length of the profile of interest
    n1 = 2/f_min * sample_rate_hz
    nperseg = min(2**np.ceil(np.log2(n1)), len_series) #compute the minimum number of samples needed to get fmin
    if nperseg == len_series:
        #just compute regular periodogram
        freqs, psd = periodogram(accelerations, sample_rate_hz)
        return car.inverse_transfer_function(freqs, psd, np.mean(velocities))
    else:
        if len_series % nperseg != 0:
            accelerations = np.concatenate((accelerations, np.zeros(len_series % nperseg)))
        freqs, ts, stf = stft(accelerations, fs=sample_rate_hz, window=window_function,
                              nperseg=nperseg, noverlap=int(overlap*nperseg), padded=True)
        freq_times, psd = average_stft_psd(car, times, ts, freqs, stf, velocities, f_min)


def compute_inverse_tf_constant_veloc(car, acc_freqs, acc_psd, veloc):
    #method for converting frequencies if the power spectral density if the car was driving at a constant velocity
    return car.inverse_transfer_function(acc_freqs, acc_psd, veloc)

def compute_inverse_tf_nonconstant_veloc(car, acc_freqs, acc_psd, velocities):
    #method for converting frequencies if the power spectral density if the car was driving at a constant velocity
    return None, None 
    #return car.inverse_transfer_function(acc_freqs, acc_psd, veloc)

def compute_roughness_coef(est_freqs, est_psd, psd_type, prof_len):
    #print("max freq, min freq are {0}, {1}".format(min(est_freqs), max(est_freqs)))
    min_freq = 1/(prof_len/2)
    if psd_type != 'welch':
        
        smth_freqs, smth_psd = cisr.smooth_psd(est_freqs, est_psd)
       
    else:
        smth_freqs, smth_psd = est_freqs, est_psd
    regress = cisr.fit_smooth_psd(smth_freqs[np.where(smth_freqs >= min_freq)], smth_psd[np.where(smth_freqs >= min_freq)])
    #regress = cisr.fit_smooth_psd(smth_freqs, smth_psd)
    est_gn0 = regress.coef_[0][0] *  1e6/.01
    #plt.loglog(est_freqs, est_psd, label='Estimated PSD')
    #plt.loglog(smth_freqs, smth_psd, label='Smoothed PSD')
    #preds = regress.coef_[0][0]* np.arange(.001, 10, .01)**-2
    #plt.loglog(np.arange(.001, 10, .01), preds.flatten(),  label='Regressed PSD')
    #plt.vlines(min_freq, 0, 100)
    #plt.legend()
    #plt.title("Smoothed vs. Estimated PSD")
    #plt.show()
    return est_gn0

def get_car_list():
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

def get_non_param_window_funcs():
    return set(['boxcar', 'triang', 'blackman', 'hamming', 'hann',
                'bartlett', 'flattop', 'parzen', 'bohman', 'blackmanharris', 'nuttall', 'barthann'])

def compute_road_psd(psd_type, window_func, window_func_param, profile, sample_rate):
    if psd_type == 'periodogram':
        f, psd = periodogram(profile.get_elevations()/1000, sample_rate)
    else:
        nperseg = 2**np.ceil(np.log2(2 * 100 * sample_rate))
        if window_func in get_non_param_window_funcs():
            f, psd = welch(profile.get_elevations(), fs=sample_rate, nperseg=nperseg, window=window_func)
        else:
            f, psd = welch(profile.get_elevations(), fs=sample_rate, nperseg=nperseg, window=(window_func, window_func_param))
    #elif psd_type == 'stft':
    #    f, psd = None, None
    #    pass # not implemented yet
    #else:
    #    print("Invalid psd method input, so defaulting to periodogram")
    #    f, psd = periodogram(profile.get_elevations()/1000, sample_rate)
    return f, psd

def get_road_class(est_gn0):
    if est_gn0 <= 32:
        return 'A'
    elif est_gn0 <= 128:
        return 'B'
    elif est_gn0 <= 512:
        return 'C'
    elif est_gn0 <= 2048:
        return 'D'
    elif est_gn0 <= 8192:
        return 'E'
    elif est_gn0 <= 32768:
        return 'F'
    elif est_gn0 <= 131072:
        return 'G'
    else:
        return 'H' 

def compute_acc_psd(psd_type, window_func, window_func_param, accelerations, sample_rate, velocities):
    if psd_type == 'periodogram':
        f, psd = periodogram(accelerations, sample_rate)
    elif psd_type == 'welch':
        veloc = np.mean(velocities)
        nperseg = 2**np.ceil(np.log2(2 * 100/veloc * sample_rate))
        if window_func in get_non_param_window_funcs():
            f, psd = welch(accelerations, fs=sample_rate, nperseg=nperseg, window=window_func)
        else:
            f, psd = welch(accelerations, fs=sample_rate, nperseg=nperseg, window=(window_func, window_func_param/veloc))
    elif psd_type == 'stft':
        f, psd = None, None
        #not implemented yet
    else:
        logging.warning("Invalid psd method input, so defaulting to periodogram")
        #print("Invalid psd method input, so defaulting to periodogram")
        f, psd = periodogram(accelerations, sample_rate)
    return f, psd

def add_row_dict(data_dict, rtype, gn0, l, vel, car_name, rmse, est_gn0, est_road_class, sr, psd_type):
    data_dict['Road_Type'].append(rtype)
    data_dict['Gn0'].append(gn0)
    data_dict['Length'].append(l)
    data_dict['Velocity'].append(vel)
    data_dict['Car'].append(car_name)
    data_dict['RMSE'].append(rmse)
    data_dict['Est_Gn0'].append(est_gn0)
    data_dict['Pred_Class'].append(est_road_class)
    data_dict['Sample_Rate_Hz'].append(sr)
    data_dict['PSD_Type'].append(psd_type)
    
def find_final_pos(p0, v0, accs, delta_t=.001):
    #will fix the velocity series if negative velocities were encountered
    v_final, p_final = v0, p0
    for acc in accs:
        delta_p = delta_t * v_final + .5 * delta_t ** 2 * acc
        delta_v = acc * delta_t
        if delta_p < 0:
            delta_p = 0
        #if delta_v < 0:
        #    delta_v = 0
        p_final = p_final + delta_p
        v_final = max(v_final + delta_v, 0) #can't have negative velocity, that would be very bad
        
    return p_final, v_final

def compute_acc_times(v_type, length, vel, acc, acc_dist):
    acc_times, v0 = [], vel #default case is constant, make this the condition if none of the below are specified
    #if v_type == 'constant': #just ignore acc parameter here
    #    return [], vel
    if v_type == 'acc_beg': #assume this means v0 of 0, and acceleration until reach each v
        time_to_reach_v = vel/acc #how long it will take to reach that velocity, initial velocity is 0
        acc_times = [(acc, time_to_reach_v)] #Note - might not reach velocity if profile isn't long enough
        v0 = 0
    elif v_type == 'acc_end': #assume this means initial v0, and acceleration until stop - which should happen where? 
        v0 = vel
        acc = -acc #always assume acc input as positive
        time_to_reach0 = -v0/acc
        #keep constant velocity until distance reached
        dec_pos = length - v0*time_to_reach0 - .5*acc*time_to_reach0**2
        const_time = dec_pos/v0
        acc_times = [(0, const_time), (acc, time_to_reach0)]
    elif v_type == 'acc_random': #assume this means initial v0, and random accelerations
        #in this case, the acc is the variance of the normal distribution
        v0 = vel
        dist = norm(loc=0, scale=acc_dist)
        #just give a ridic amount of time, go until reach end of profile
        acc_times = (dist, 10000) #hopefully doesn't reach 0, because probably will be hard to escape 0
    elif v_type == 'acc_beg_end': #assume this means starting from 0, accelerate to v, then stop (at same rate)
        time_to_reach_v = vel/acc
        v0 = 0
        pos = .5*acc*time_to_reach_v**2
        time_to_reach0 = -v0/(-acc) #assume acc is positive

        dec_pos = length - vel*time_to_reach0 - .5*(-acc)*time_to_reach0**2 #assume acc is positive
        if dec_pos > pos: #need to keep velocity at for constant time 
            const_time = (dec_pos - pos)/vel
        else:
            const_time = 0
         
        acc_times = [(acc, time_to_reach_v), (0, const_time), (-acc, time_to_reach0)]
    elif v_type == 'acc_beg_rand': #assume this means starting from 0, accelerate to v, then random
        v0 = 0
        time_to_reach_v = vel/acc

        rand_accs = norm(loc=0, scale=acc_dist).rvs(50000) #I don't like this - simulate data for 50 seconds, hope that reaches end
        acc_times = [(acc, time_to_reach_v)] +  list(zip(rand_accs, len(rand_accs)*[.1]))
    elif v_type == 'acc_rand_end': #assume this means starting from v0, then random, then decelerate to stop
        acc = -acc #assume acc is positive
        v0 = vel
        rand_accs = norm(loc=0, scale=.01).rvs(10000) #random for 10 seconds
        p_final, v_final = find_final_pos(0, v0, rand_accs)
        five_acc_time = 0
        if v_final == 0: #probably won't happen... but is possible
            #accelerate up to 5 m/s**2
            five_acc_time = 1
            p_final = p_final + .5*five_acc_time**2
            v_final = 5
            #possible that p_final could now be greater than end of profile, but if so, I don't care
        time_to_reach0 = -v_final/(acc) #assume acc positive always
        dec_pos = length - v_final*time_to_reach0 - .5*acc*time_to_reach0**2
        if dec_pos > p_final: #need to keep velocity at for constant time 
            const_time = (dec_pos - p_final)/vel
        else:
            const_time = 0
        acc_times = [(rand_accs, 10), (5, five_acc_time), (0, const_time), (acc, time_to_reach0)]
    elif v_type == 'acc_beg_rand_end': #final case, start from 0, accelerate to v, 
        v0 = 0
        time_to_reach_v = vel/acc

        rand_accs = norm(loc=0, scale=.01).rvs(10000) #random for 10 seconds
        p_final, v_final = find_final_pos(.5*acc*time_to_reach_v**2, v0, rand_accs)
        five_acc_time = 0
        if v_final == 0: #probably won't happen... but is possible
            #accelerate up to 5 m/s**2
            five_acc_time = 1
            p_final = p_final + .5*five_acc_time**2
            v_final = 5
        time_to_reach0 = -v_final/(-acc) #assume acc positive always
        dec_pos = length - v_final*time_to_reach0 - .5*acc*time_to_reach0**2
        if dec_pos > pos: #need to keep velocity at for constant time 
            const_time = (dec_pos - p_final)/vel
        else:
            const_time = 0
        acc_times = [(acc, time_to_reach_v), (rand_accs, 10), (5, five_acc_time), (0, const_time), (-acc, time_to_reach0)]
        #how to decelerate...
    return acc_times, v0


  


def run_cars_over_profile(cars, rtype, l, vel, acc, sr,  v_type,  acc_dist, psd_type, window, wparam, 
                            dict_params,road_freqs, road_psd, use_c, gn0, curr_profile, ret_dict=None):
    if ret_dict is None:
        ret_dict = {}
        for pname in dict_params:
            ret_dict[pname] = []
    avg_psd = None
    for car_tup in cars: #return a different row for average of cars at a velocity, and all cars irrespective of velocity
        car_name, car = car_tup[0], car_tup[1]
        acc_times, v0 = compute_acc_times(v_type, l, vel, acc, acc_dist)
        T, yout, xout, new_dists, new_els, velocities = car.run2(curr_profile, acc_times, v0,
                                                        final_sample_rate=sr)
        f_acc, psd_acc = compute_acc_psd(psd_type, window, wparam, yout[:, -1],
                                                                 sr, velocities)
        if use_c:
            f_road_est, psd_road_est = compute_inverse_tf_constant_veloc(car, f_acc, psd_acc, np.mean(velocities))
        else:
            f_road_est, psd_road_est = compute_inverse_tf_nonconstant_veloc(car, f_acc, psd_acc, velocities)
        f_final, errors = rmse_frequency_bins(road_freqs, road_psd, f_road_est, psd_road_est)
        rmse = np.sum(errors**2/len(errors))
        #plt.loglog(road_freqs, road_psd, label='Road PSD')
        #plt.loglog(np.arange(.01, 10, .01), mp.iso_psd_function(gn0, np.arange(.01, 10, .01)), label='Hypothetical PSD')
        est_gn0 = compute_roughness_coef(f_road_est, psd_road_est, psd_type, l)
                                
        est_road_class = get_road_class(est_gn0)
        add_row_dict(ret_dict, rtype, gn0, l, vel, car_name, rmse, est_gn0, est_road_class, sr, psd_type)
        if avg_psd is None:
            avg_psd = psd_road_est
        else:
            avg_psd += psd_road_est
        
    avg_psd = avg_psd/len(cars)
    f_avg, e_avg = rmse_frequency_bins(road_freqs, road_psd, f_road_est, avg_psd)
    rmse_avg = np.sum(e_avg**2/len(e_avg))
    est_gn0_avg = compute_roughness_coef(f_road_est, avg_psd, psd_type, l)
    est_rt_avg = get_road_class(est_gn0_avg)
    add_row_dict(ret_dict, rtype, gn0, l, vel, 'Avg_{0}_{1}'.format(vel, sr), rmse_avg, est_gn0_avg, est_rt_avg, sr, psd_type)
    return ret_dict 

def process_from_queue(exp_queue, res_queue, is_processing, pnum=0):
    #while not lock.acquire():
    #    time.sleep(.01)
    #lock.release()
    #logging.debug("Pnum {0} starting".format(pnum))
    while is_processing.value:
        try:
            #logging.debug("Pnum {0} trying to read from queue".format(pnum))
            #curr_row = exp_queue.get(block=True, timeout=10)
            #curr_row = exp_queue.get(block=True, timeout=5) #give five seconds to wait for something to be available from queue, then quit if not available
            curr_row = exp_queue.get(block=False)
            exp_queue.task_done()
        except queue.Empty: 
            continue #go back to beginning of loop if you get this exception, because is_processing determines whether more data needs to be read
        except Exception as e:
            #continue #if exception, 
            logging.error("Received exception {0}, exiting process from queue, pid = {1}".format(e, pnum))
            return
            #return #need to set task done here?
        try:
            res = run_cars_over_profile(**curr_row)
            res_queue.put(res)
            #logging.debug("Successfully added item to res_queue")
        except Exception as e:
            logging.error('Failed experiment with row {0}'.format(curr_row))
        finally:
            logging.debug("Finished exp_queue task")
    logging.debug("Should be exiting process from queue, pid = {0}".format(pnum))
    return
    #return
            #logging.debug("Exp queue length now {0}".format(exp_queue.qsize()))
    
        


def run_experiment(args, mp_queue):
    #Set logging level
    #locked = lock.acquire() #block so can add things to loop before other processes try to read from it
    
    v_type = args.v_type
    use_c = args.use_cv
    cars  = get_car_list()
    vels = list(map(lambda x: float(x), args.vels))
    lengths = list(map(lambda x: float(x), args.lengths))
    srs = list(map(lambda x: int(x), args.sr))
    rtypes = args.road_types
    n_per_class = args.num_roads
    result_dict = {'Road_Type': [], 'Gn0': [], 'Length': [], 'Velocity': [], 'Car': [], 'RMSE': [],
                   'Est_Gn0': [], 'Pred_Class': [], 'Sample_Rate_Hz': [], 'PSD_Type': []}
    #if v_type == 'constant': #constant velocity, can just use compute_psd_constant velocity
    seed = args.seed
    logging.debug("Rtypes is {0}, n per class is {1}\n".format(rtypes, n_per_class))
    logging.debug("Lengths is {0}\n".format(lengths))
    logging.debug("Velocities is {0}\n".format(vels))
    logging.debug("Accs is {0}\n".format(args.accs))
    logging.debug("Srs is {0}\n".format(srs))
    for rtype, rnum in zip(rtypes, len(rtypes)*[n_per_class]):
        for l in lengths:
            for _ in range(0, rnum):
                dx = .05 #maybe make this tweakable in future
                dists, elevations, gn0 = mp.make_profile_from_psd(rtype, 'sine', dx, l, seed, True) #make the profile
                curr_profile = roadprofile.RoadProfile(dists, elevations)
                road_freqs, road_psd = compute_road_psd(args.psd_type, args.window, args.wparam, curr_profile, 1/dx)
                for vel in vels:
                    for acc in args.accs:
                        for sr in srs:
                            #avg_psd = None
                            if args.use_mp:
                                mp_queue.put({'cars': cars, 'rtype': rtype, 'l': l, 'vel': vel, 'acc': acc, 'sr': sr, 'v_type': v_type, 'acc_dist': args.acc_dist, 
                                'psd_type': args.psd_type, 'window': args.window, 'wparam': args.wparam,  'dict_params': list(result_dict),
                                'road_freqs': road_freqs, 'road_psd': road_psd, 'use_c': use_c, 'gn0': gn0, 'curr_profile': curr_profile})
                                #logging.debug("Put element in queue in run experiment")
                                #if locked:
                                #    lock.release()
                                #    locked = False
                                #    logging.debug("Lock has been released by run experiment")
                                #continue
                            else:
                                st_time = time.time()
                                result_dict = run_cars_over_profile(**{'cars': cars, 'rtype': rtype, 'l': l, 'vel': vel, 'acc': acc, 'sr': sr, 'v_type': v_type, 'acc_dist': args.acc_dist, 
                                'psd_type': args.psd_type, 'window': args.window, 'wparam': args.wparam, 'dict_params': list(result_dict), 
                                'road_freqs': road_freqs, 'road_psd': road_psd, 'use_c': use_c, 'gn0': gn0, 'curr_profile': curr_profile, 'ret_dict': result_dict})
                                logging.debug("Time for all cars to run over one profile was {0} seconds".format(time.time()  - st_time))
                            #for car_tup in cars: #return a different row for average of cars at a velocity, and all cars irrespective of velocity
                                
                            #    car_name, car = car_tup[0], car_tup[1]
                                
                            #    acc_times, v0 = compute_acc_times(v_type, l, vel, acc, args.acc_dist)
                            #    T, yout, xout, new_dists, new_els, velocities = car.run2(curr_profile, acc_times, v0,
                            #                                                 final_sample_rate=sr)
                            #    f_acc, psd_acc = compute_acc_psd(args.psd_type, args.window, args.wparam, yout[:, -1],
                            #                                     sr, velocities)
                            #    if use_c:
                            #        f_road_est, psd_road_est = compute_inverse_tf_constant_veloc(car, f_acc, psd_acc, np.mean(velocities))
                            #    else:
                            #        f_road_est, psd_road_est = compute_inverse_tf_nonconstant_veloc(car, f_acc, psd_acc, velocities)
                            #    f_final, errors = rmse_frequency_bins(road_freqs, road_psd, f_road_est, psd_road_est)
                            #    rmse = np.sum(errors**2/len(errors))
                                #plt.loglog(road_freqs, road_psd, label='Road PSD')
                                #plt.loglog(np.arange(.01, 10, .01), mp.iso_psd_function(gn0, np.arange(.01, 10, .01)), label='Hypothetical PSD')
                            #    est_gn0 = compute_roughness_coef(f_road_est, psd_road_est, args.psd_type, l)
                                
                            #    est_road_class = get_road_class(est_gn0)
                            #    add_row_dict(result_dict, rtype, gn0, l, vel, car_name, rmse, est_gn0, est_road_class, sr, args.psd_type)
                            #    if avg_psd is None:
                            #        avg_psd = psd_road_est
                            #    else:
                            #        avg_psd += psd_road_est
                            #avg_psd = avg_psd/len(cars)
                            #f_avg, e_avg = rmse_frequency_bins(road_freqs, road_psd, f_road_est, avg_psd)
                            #rmse_avg = np.sum(e_avg**2/len(e_avg))
                            #est_gn0_avg = compute_roughness_coef(f_road_est, avg_psd, args.psd_type, l)
                            #est_rt_avg = get_road_class(est_gn0_avg)
                            #add_row_dict(result_dict, rtype, gn0, l, vel, 'Avg_{0}_{1}'.format(vel, sr), rmse_avg, est_gn0_avg, est_rt_avg, sr, args.psd_type)


                seed += 1
    if not args.use_mp:
        df = pd.DataFrame(result_dict)
        df.to_csv('{0}.csv'.format(args.outfile))
    #else:
    #    mp_queue.close()
    return



#if __name__ == 'main':
#    print("NAME IS MAIN")


def save_to_file(output_queue, keep_writing, filename):
    counter = 0
    with open("{0}.csv".format(filename), 'w', newline='') as csv_file:
        csvwriter = csv.writer(csv_file)
        while keep_writing.value:
            try:
                new_res = output_queue.get(block=False)
                output_queue.task_done()
                if counter == 0:
                    csvwriter.writerow(list(new_res))
                    #final_dict = new_res
                for row in zip(*list(new_res.values())):
                    #logging.debug("Row is {0}".format(row))
                    csvwriter.writerow(list(row))
                    logging.debug("Writing row {0} to file".format(counter))
                #else:
                #    for k in final_dict:
                #        final_dict[k] += new_res[k]
            except queue.Empty:
                #logging.debug("Queue is empty, keep writing is {0}".format(keep_writing.value))
                continue #using keep writing 
            except Exception as e:
                logging.error("Received exception {0} when writing to file".format(e))
                break #assume if exception, got everything from queue
            counter += 1
    #df = pd.DataFrame(final_dict)
    #df.to_csv('{0}.csv'.format(filename))
    logging.debug("Outside of loop, should be done saving to file")
    return

def parse_args_and_run():
    
    parser = argparse.ArgumentParser(
        description="""Program for running experiment to see how well we can reproduce original road profile PSD from 
                    acceleration PSD under varying velocities, paramter uncertainties, etc using transfer function """)
    #parser.add_argument('-A', '--A_profiles', '--num_A', nargs='?', type=int, default=0, const=0, metavar='N',
    #                    help='The number of class A road profiles to generate (per specified length)')
    #parser.add_argument('-B', '--B_profiles', '--num_B', nargs='?', type=int, default=0, const=0, metavar='N',
    #                    help='The number of class B road profiles to generate (per specified length)')
    #parser.add_argument('-C', '--C_profiles', '--num_C', nargs='?', type=int, default=0, const=0, metavar='N',
    #                    help='The number of class C road profiles to generate (per specified length)')
    
    parser.add_argument('--road_types', '--rts', '--road_class', '--road_classes',  nargs='*',
                        metavar='[A, B, C, .... ]',
                        help='The types of roads to generate')
    parser.add_argument('--log_level', '--log_lev', '--loglevel', '--log',  nargs='?',
                        default='WARNING',
                        const='WARNING',
                        metavar='DEBUG INFO WARNING ERROR',
                        help='The level of logging to use')
    parser.add_argument('--num_roads', '--n_roads', '--total_roads', '--roads_per_class', nargs='?', type=int,
                        default=1,
                        const=1,
                        metavar='N',
                        help='The numer of roads (per class) to generate')
    #parser.add_argument('-', '--C_profiles', '--num_C', nargs='?', type=int, default=0, const=0, metavar='N',
    #                    help='The number of class C road profiles to generate (per specified length)')
    parser.add_argument('--lengths', '--p_lengths', '--profile_lengths',  nargs='*',
                        metavar='[len_1, len_2, ...., len_n]',
                        help='The lengths of the profiles at which to compute the experiments at')
    parser.add_argument('--vels', '--velocities', '--velocity', nargs='*',
                        metavar='[vel_1, vel_2, ..., vel_n]',
                        help='The velocities at which to run the simulations at')

    #parser.add_argument('--v_max', '--vel_max', '--velocity_max', nargs='?', type=float,
    #                    default=40.0,
    #                    const=40.0,
    #                    metavar='X.x',
    #                    help='The maximum velocity at which to compute the experiment')
    parser.add_argument('--outfile', '--out_file', '--title', nargs='?',
                        default='/Users/gregoryislas/Documents/Mobilized/data_dump/PSD_Exp/unknown.csv',
                        const='/Users/gregoryislas/Documents/Mobilized/data_dump/PSD_Exp/unknown.csv',
                        metavar='file_path/experiment_title',
                        help='The name of the output file, supposed to be used as descriptor for parameters of experiment ')
    #parser.add_argument('--car', '--car_type', nargs='?',
    #                    default='car1',
    #                    const='car1',
    #                    metavar='[car_type]',
    #                    help='The type of car to use for driving over the profile')
    parser.add_argument('--v_type', '--velocity_type',  nargs='?',
                        default='constant',
                        const='constant',
                        metavar='[constant, acc_beg, acc_end, acc_random, acc_both]',
                        help='Specified the velocity of the car as it drives over the road profile. Also used as initial' +
                             'velocities when non constant acceleration is input ')
    #parser.add_argument('--v0', '--init_vel', '--initial_velocity', nargs='?', type=float,
    #                    default=0.0,
    #                    const=0.0,
    #                    metavar='X.x',
    #                    help='The initial velocity of the car. Only used if non-constant velocity specified')
    parser.add_argument('--accs', '--accelerations', nargs='*',
                        default=[1],
                        metavar='[acc_1, acc_2, ...., acc_n]',
                        help='Specifies the list of accelerations to try if constant acceleration chosen. Otherwise, ' +
                        'specifies the variances of the acceleration distributions to try')
    #parser.add_argument('--acc_var', 'acc_variance', nargs='?', type=float,
    #                    default=.01,
    #                    const = .01,
    #                    metavar='X.x',
    #                    help='The standard deviation of the acceleration distribution (only used if acc_random specified)')
    #parser.add_argument('--v_min', '--vel_min', '--velocity_min',  nargs='?', type=float,
    #                    default=5.0,
    #                    const=5.0,
    #                    metavar='X.x',
    #                    help='The minimum velocity at which to compute the experiment')
    #parser.add_argument('--v_max', '--vel_max', '--velocity_max', nargs='?', type=float,
    #                    default=40.0,
    #                    const=40.0,
    #                    metavar='X.x',
    #                    help='The maximum velocity at which to compute the experiment')
    parser.add_argument('--sr', '--sample_rates', '--sample_rates_hz', '--hertz', '--hz', nargs='*',

                        metavar='[sr_1, sr_2, ..., sr_n]',
                        help='The sample rates (in the time domain) at which to sample the acceleration series. ' +
                             'Maximum of 1000 Hz')
    parser.add_argument('--use_cv', '--use_constant_veloc', '--use_constant_velocity',  nargs='?', type=int,
                        default=1,
                        const=1,
                        metavar='1/0',
                        help='Whether or not to use a constant velocity model')
    parser.add_argument('--psd_type', '--psd', nargs='?',
                        default='periodogram',
                        const='periodogram',
                        metavar='XXX',
                        help='The methodology of computing the Power Spectral Density. Defaults to periodogram. If welch or STFT input, must also specify window function')
    parser.add_argument('--window', '--window_func', '--window_function', nargs='?',
                        default='hann',
                        const='hann',
                        metavar='[hann, boxcar, triang, blackman, hamming, bartlett, flattop, parzen, bohman, blackmanharris, ' +
                                'nuttall, barthann, kaiser, gaussian, slepian, dpss, chebwin, exponential, tukey]',
                        help='The window function. If specify a window function that needs a parameter, must also speficy that')
    parser.add_argument('--wparam', '--window_param', '--window_parameter', '--w_param', '--window_func_param',
                        '--window_function_param', nargs='?', type=float,
                        default=0.1,
                        const=0.1,
                        metavar='X.x',
                        help='The window function parameter. If you don\'t specify this, it will use .1, which is probably not what you want')
    parser.add_argument('--acc_dist', '--acc_scale', '--acc_dist_scale',  nargs='?', type=float,
                        default=0.05,
                        const=0.05,
                        metavar='X.x',
                        help='The variance for the normal distribution of the accelerations. If you don\'t specify this, it will use .1, which is probably not what you want')
    parser.add_argument('--use_mp', '--mp', '--multi_process', '--multi_thread', nargs='?', type=int,
                        default=0,
                        const=0,
                        metavar='0/1', 
                        help='Whether or not to use multiprocessing')
    parser.add_argument('--num_threads', '--num_workers', '--threads', '--processes', nargs='?', type=int,
                        default=1,
                        const=1,
                        metavar='X', 
                        help='The number of processes to spawn for the multiprocessing')
    parser.add_argument('--seed', '--seed_start', nargs='?', type=int,
                        default=1,
                        const=1,
                        metavar='X', 
                        help='The starting seed to use for the experiment')
    args = parser.parse_args()
    exp_queue, result_queue = None, None
    st_time = time.time()
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
   
    if args.use_mp and args.num_threads > 1:
        #rtypes, lengths, nroads = args.road_types, args.lengths, args.num_roads
        #Split up for multiprocessing
        exp_queue, result_queue = multiprocessing.JoinableQueue(),  multiprocessing.JoinableQueue()
        #total_profs = len(rtypes)*len(lengths)*nroads
        #n_threads = args.num_threads//2 + args.num_threads%2 #use half of the processes to add to the queue
        #exp_procs = []
        #if n_threads <= len(rtypes): #ideal case
        #    n_per_thread = len(rtypes)//n_threads
        #    st_index = 0
        #    for x in range(0, n_threads):
        #        if x == n_threads - 1:
        #            args.road_types = rtypes[st_index:]
        #        else:
        #            args.road_types = rtypes[st_index:st_index + n_per_thread]
        #        logging.debug("args rtypes is {0}".format(args.road_types))
        #        exp_proc = multiprocessing.Process(target=run_experiment, args=(args, exp_queue))
        #        exp_proc.start()
        #        exp_procs.append(exp_proc)
        #        st_index = st_index + n_per_thread
        #elif n_threads <= len(lengths): #next best scenario
        #    n_per_thread = len(lengths)//n_threads
        #    st_index = 0
        #    for x in range(0, n_threads):
        #        if x == n_threads - 1:
        #            args.lengths = lengths[st_index:st_index + n_per_thread]
        #        else:
        #            args.lengths = lengths[st_index:]
        #        exp_proc = multiprocessing.Process(target=run_experiment, args=(args, exp_queue))
        #        exp_proc.start()
        #        exp_procs.append(exp_proc)
        #        st_index = st_index + n_per_thread
        #elif n_threads <= nroads: #still okay
        #    n_per_thread = nroads//n_threads
        #    st_index = 0
        #    for x in range(0, n_threads):
        #        if x == n_threads - 1:
        #            args.num_roads = nroads - st_index
        #        else:
        #            args.num_roads = n_per_thread
        #        exp_proc = multiprocessing.Process(target=run_experiment, args=(args, exp_queue))
        #        exp_proc.start()
        #        exp_procs.append(exp_proc)
        #        st_index = st_index + n_per_thread
        #else: #Ummm - very not ideal, will have to figure out how to split them later
        #    logging.error("Please use the single threaded, I didn't implement the case where your number of threads is higher than the lengths, number of classes, and road types")
        #    sys.exit(1)
            #n_per_thread = total_profs//n_threads
            ##rt_index, le, nr_index = 0, 0, 0
            #rt_end, le_end, nr_end = 0, 0, 0
            #inc_rt, inc_lens, inc_nrs = True, True, True
            #for x in range(0, n_threads):
            #    if inc_rt:
            #        rt_end = rt_index + 
            #    if x == n_threads - 1:
            #        pass
            #    else:
            #        args.rtypes = rtypes[rt_index:rt_index + len(rtypes)//n_per_thread]
            #        args.lengths = lengths[le:le + len(lengths)//n_per_thread]
            #        args.num_roads = nr_index + nroads//n_per_thread
            #    rt_index = (rt_index + (len(rtypes)//n_per_thread))%len(rtypes)
            #    le  = (le + (len(lengths)//n_per_thread))%len(rtypes)
            #    nr_index = (nr_index + nroads//n_per_thread)

            #while rt_index < len(rtypes):
            #    args.rtypes = rtypes[rt_index:rt_index + len(rtypes)//n_per_thread]
            #    nr = 0
            #    while nr < nroads:
            #        le = 0
            #        while le < len(lengths):
            #            args.lengths = lengths[le:le + n_threads]
            #            exp_proc = multiprocessing.Process(target=run_experiment, args=(args, exp_queue))
            #        args.num_roads = nr 
            
        exp_proc = multiprocessing.Process(target=run_experiment, args=(args, exp_queue))
        exp_proc.start()
        time1 = time.time()
        logging.debug("Starting run experiment")
        #lock = multiprocessing.Lock()
        #exp_proc = multiprocessing.Process(target=run_experiment, args=(args, exp_queue, ))
        #run_experiment(args, exp_queue)
        #exp_proc.start() #so can add to queue and read from it all at once
        #time.sleep(1)
        
        #logging.debug("To add everything to queue, took {0} seconds".format(time.time() - time1))      
        proc = []
        pnum = 1
        #exp_queue.join()
        logging.debug("Starting processes to read from queue")
        val =  multiprocessing.Value('i', 1)
        for _ in range(0, args.num_threads - 1):
            p = multiprocessing.Process(target=process_from_queue, args=(exp_queue, result_queue,  val, pnum))
            p.start()
            proc.append(p)
            pnum += 1
        logging.debug("Finished starting all reading processes, now waiting for all queue inserting processes to finish")
        #while True:
        #    for x in range(len(exp_procs)-1, -1, -1):
        #        ep = exp_procs[x]
                
        #        if not ep.is_alive() or ep.exitcode is not None:
        #            logging.debug("About to join")
        #            ep.join()
        #            logging.debug("sucessfully joined")
        #            exp_procs.pop(x)
                    #now, start another reading process as well
        #            p = multiprocessing.Process(target=process_from_queue, args=(exp_queue,result_queue,  val, pnum))
        #            p.start()
        #            proc.append(p)
        #            pnum += 1
        #    if len(exp_procs) == 0:
        #        break
        #now all inserting processes have finished, wait for all reading processes to finish
        
        while exp_proc.is_alive():
            time.sleep(.01)
        exp_proc.join()
        #p_final = multiprocessing.Process(target=process_from_queue, args=(exp_queue,result_queue, val, pnum))
        #p_final.start()
        #proc.append(p_final)
        logging.debug("Starting reading from queue process")
        keep_writing = multiprocessing.Value('i', 1)
        last_process = multiprocessing.Process(target=save_to_file, args=(result_queue, keep_writing, args.outfile))
        last_process.start()
        logging.debug("All inserting processes have finished, joining exp_queue")
        exp_queue.join() #wait for everything to be done reading from the queue
        val.value = 0 #now, can stop processing tasks safely
        #    time.sleep(.01)
        #while exp_proc.is_alive():
        #    time.sleep(.01)
        #logging.debug("To add everything to queue, took {0} seconds".format(time.time() - time1))
        #exp_proc.join()
        
        #logging.debug("Before queue join")
        #exp_queue.join()
        #logging.debug("After queue join")
        for p in proc:
            logging.debug('is alive {0}'.format(p.is_alive()))
            if p.is_alive():
                p.join() 
        
        logging.debug("After process join")
        result_queue.join() #wait for everything to be processed here
        logging.debug("result queue joined")
        keep_writing.value = 0
        last_process.join()
        logging.debug("last process joined")
    else:
        run_experiment(args, None)
    logging.info("Total program time was {0} seconds".format(time.time()  - st_time))
    

    #title of csv file is Experiment parameters
#parse_args_and_run()
