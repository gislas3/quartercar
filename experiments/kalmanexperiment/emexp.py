import sys, time, argparse, logging, pickle, gzip, os
import numpy as np
import pandas as pd
from multiprocessing import Pool, TimeoutError
from matplotlib import pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_continuous_white_noise
#from experiments import computeisoroughness as cisr
from quartercar import roadprofile, qc, qckalman, cars

from scipy.linalg import expm
from math import factorial


####IDEA - Use QC model along with ramp inputs to help simulate data to see how well kalman filter for orientation estimation works

def initialize_kf_est_params(true_car, true_road_sigma, F_known, Q_known, ku_mu, sigma, dt, order=1):
    """
    This function will initialize the kalman filter parameters for the kalman filtering round where the objective
    is to estimate the unknown model parameters, e.g. if you want to estimate the car parameters, the process noise matrix, etc.
    
    Assumes a 4-dimensional state space of: x_dot_s (sprung mass velocity in units m/s), 
                                            x_s - y (sprung mass displacement - road profile elevation in units m), 
                                            x_dot_u (unsprung mass velocity in units m/s), 
                                            x_u - y (unsprung mass displacement - road profile elvation in units m)

    Assumes a 1-dimensional measurement of: x_dot_dot_s (unsprung mass acceleration in units m/s**2)
    
    :param true_car: An instance of `QC` which defines the "true car" that drove over the road profile
    :param true_road_sigma: The "true" value of the variance of the road profile in the time domain.
    :param Q_known: A flag (assumed to be either 0 or 1) which denotes whether Q will be known when running the kalman filter.
    :param F_known: A flag (assumed to be either 0 or 1) which denotes whether F will be known when running the kalman filter.
    :param ku_mu: Represents the value of ku/mu (tire spring stiffness divided by unsprung mass in SI units). If None, will use the value of ku_mu from true_car (so None means it should be known)
    :param sigma: Represents the variance of the measurement noise (accelerometer noise)
    :param dt: Represents the spacing of the measurements in the time domain (in seconds)
    :param order: Represents the order of the euler approximation for the transition matrix. Default: 1
    :return: F_final (state transition matrix), H (measurement matrix), Q (process noise covariance matrix), R (measurement noise covariance matrix), M (constraint matrix s.t. M * F = G)
    G (constraint matrix s.t. M * F = G), p_to_opt (the parameters to optimize over in the EM algorithm)
    """
    p_to_opt = ['x0', 'P0'] # Keeps track of parameters to optimize over. x0 and P0 are added by default, since in practice we will never know these (athough idea: perhaps can measure a speedbump outside the depot, then look for that in data....)
    if Q_known:
        road_sigma = true_road_sigma
    else: # Going to assume that we are not off by more than 100% when we randomly choose the road profile sigma
        e_road_sigma = np.random.uniform(0, 1)
        road_sigma =  (1 + np.random.choice([1, -1]) * e_road_sigma ) * road_sigma
        p_to_opt.append('Q')

    Q = np.array([[0, 0, 0, 0], 
        [0, road_sigma, 0, road_sigma], 
        [0, 0, 0, 0],
        [0, road_sigma, 0, road_sigma]]) #Initialize Q, process noise covariance matrix
    # (1-e)*road_sigma = est_sigma - road_sigma
    #higher: e = .2: est_sigma = 1.2 * road_sigma, est_sigma - road_sigma  = 1.2road_sigma - road_sigma = road_Sigma(1.2  - 1) = .2*road_sigma
    #lower: e = .2: est_sigma = .8 * road_sigma; est_sigma - road_sigma = road_sigma(.8 - 1) = -.2 * road_sigma
    if F_known: #TODO: Implement higher orders later
        m_s = true_car.m_s
        c_s = true_car.c * true_car.m_s        #self.c = c_s/self.m_s
        k_u = true_car.k1 * true_car.m_s        #self.k1 = k_u/self.m_s
        k_s = true_car.k2 * true_car.m_s
        m_u = true_car.mu * true_car.m_s       #self.k2 = k_s/self.m_s
        cs_ms = c_s/m_s
        ks_ms = k_s/m_s
        ku_mu = k_u/m_u
        cs_mu = c_s/m_u
        ks_mu = k_s/m_u
        ku_ks_mu = (k_u + k_s)/m_u
    else:
        #randomly select parameters?
        # These values are taken from Page 938 of Jazar's Vehicle Dynamics textbook, however the units he used I believe are in lbs/inches, so I converted to SI units
        # eps: 1 to 20 = ms/mu
        # ks/ms: 4**2 to 20 ** 2
        # ku/mu: 40**2 to 400 **2
        # cs/ms: 0.4 to 40 #40?!
        # cs/mu: cs/ms * eps 
        # ks/mu: ks/ms * eps 
        # (ku + ks)/mu: ks/mu + ku/mu
        if ku_mu is None:
            m_s = true_car.m_s
            k_u = true_car.k1 * true_car.m_s  
            m_u = true_car.mu * true_car.m_s
            ku_mu = k_u/m_u
        
        
        eps = np.random.uniform(1, 20)
        ks_ms = np.random.uniform(4**2, 20**2)
        
        cs_ms = np.random.uniform(.4, 40)
        cs_mu = cs_ms * eps
        ks_mu = ks_ms * eps
        ku_ks_mu = ku_mu + ks_ms
            #ku_ks_mu = ks_ms + ku_mu 
        p_to_opt.append('F')
        #cs_mu = np.random.uniform(.4, 20*40)
        #ks_mu = 
        
        
    #else:
    #    ku_mu = np.random.uniform(40**2, 400**2)

    
    
    #F = np.array([[-1*c_s/m_s, -1*k_s/m_s, c_s/m_s, k_s/m_s], 
    #    [1, 0, 0, 0], 
    #    [c_s/m_u, k_s/m_u, -1*c_s/m_u, -1*(k_u + k_s)/m_u], 
    #    [0 , 0, 1, 0]])
    F = np.array([[-1*cs_ms, -1*ks_ms, cs_ms, ks_ms], 
        [1, 0, 0, 0], 
        [cs_mu, ks_mu, -1*cs_mu, -1*ku_ks_mu], 
        [0 , 0, 1, 0]]) # initialize F
    F_final = np.eye(len(F))
    for x in range(1, order + 1): # taylor series approximation
        F_final += F*(dt**x)/factorial(x)

    R = np.array([[sigma]]) # Only one measurement (sprung acc). Thanks a lot Kevin.
    #H = np.array([[-1*c_s/m_s, -1*k_s/m_s, c_s/m_s, k_s/m_s]])
    H = np.array([[-1*cs_ms, -1*ks_ms, cs_ms, ks_ms]])
    M = np.array([[1, 0], [0, 1], [1, 0], [0, 1]]) # Constraint matrix
    G = np.array([[1, 0], [dt, 1], [1, -1*ku_mu*dt], [dt, 1]]) # Other constraint matrix
    return F_final, H, Q, R, M, G, p_to_opt



# def add_est_car_params(return_dict, dt, F_final, H_final, Q_final, R_final, x0_final, P0_final, x_smoothed_final, P_smoothed_final, log_l):
#     """
#     This is a handy helper function to add parameters to the return dictionary. UPDATE: This is no longer used in the code...
#     :param return_dict: The dictionary which will store all these values
#     :param dt: The time spacing between measurements
#     :param F_final: The estimated state transition matrix
#     :param H_final: The estimated measurement matrix.
#     :param Q_final: The estimated process noise covariance matrix.
#     :param R_final: The estimated measurement noise covariance matrix.
#     :param x0_final: The estimated initial state.
#     :param P0_final: The estimated initial state covariance.
#     :param x_smoothed_final: The estimated sequence of states produced by the kalman filter.
#     :param P_smoothed_final: The estimated sequence of covariances of the states produced by the kalman filter.
#     :param log_l: The log likelihood of the state sequence under the model
#     """
    
#     return_dict['F_Final'] = F_final
#     return_dict['Est_Ks_Ms'] = F_final[0][3]/dt # ks/ms * dt should be F[0][3] (1st row, 4th column)
#     return_dict['Est_Cs_Ms'] = F_final[0][2]/dt # cs/ms * dt should be F[0][2] (1st row, 3rd column)
#     return_dict['Est_Cs_Mu'] = F_final[2][0]/dt # cs/mu * dt should be F[2][0] (3rd row, first column)
#     return_dict['Est_Ks_Mu'] = F_final[2][1]/dt # ks/mu * dt should be F[2][1] (3rd row, second column)
#     return_dict['Est_Ku_Mu'] = (F_final[2][3] + F_final[2][1])/dt # Check this
#     #return_dict['Est_Ks_Ms'] = F_final
#     return_dict['Q_Final'] = Q_final # 
#     #return_dict['X0_final'] = x0_final
#     #return_dict['P0_final'] = P0_final
#     #return_dict['X_Smoothed'] = x_smoothed_final
#     #return_dict['P_Smoothed'] = P_smoothed_final
#     return_dict['Log_L'] = log_l

def add_true_car_params(ret_dict, cname, true_car):
    """
    Handy helper function for adding the true car parameters to the data dictionary
    :param ret_dict: The dictionary which contains the data from the simulation
    :param cname: The name of the car which is driving over the road profile.
    :param true_car: A `QC` instance which contains the information and handy methods about the car that drove/will drive over the profile.
    
    """
    ret_dict['Car'].append(cname)
    ret_dict['True_Eps'].append(1/true_car.mu)
    ret_dict['True_Ws'].append(np.sqrt(true_car.k2))
    ret_dict['True_Wu'].append(np.sqrt(true_car.k1/true_car.mu))
    ret_dict['True_Xi'].append(true_car.c/(2*np.sqrt(true_car.k2)))
    #ret_dict['True_Ms'] = true_car.m_s
    #ret_dict['True_Mu'] = true_car.m_s * true_car.mu
    #ret_dict['True_Ks'] = true_car.k2 * true_car.m_s
    #ret_dict['True_Ku'] =  true_car.k1 * true_car.m_s 
    #ret_dict['True_Cs'] = true_car.c * true_car.m_s 

def reformat_params(kf_em_params, dt):
    """
    A handy helper function that will allow us to extract the proper parameter information from the kalman filter output.
    :param kf_em_params: The output from the EM algorithm of the kalman filter.
    :param dt: The spacing in the time domain between measurements.
    :return: eps (ms/mu), ws (sqrt(ks/ms)), wu (sqrt(ku/mu)), xi (cs/(2 * ms * ws)), Q (process noise covariance), 
    x0 (initial state), P0 (initial state covariance), x_smoothed (sequence of smoothed states produced by kalman filter), 
    P_smoothed (sequence of smoothed covariance estimates produced by kalman filter)
    """
    # Assuming (probably not a good idea) that the order of kf_em_params will be:
    # F, H, Q, R, x0, P0, x_smoothed, P_smoothed, log_l
    F = kf_em_params[0] # 
    Q = kf_em_params[2]
    eps = F[2][0]/F[0][2] # ks/mu/(ks/ms) = ks/mu * ms/ks = ms/mu
    ws = np.sqrt(F[0][3]/dt) # 1st row, fourth column should be ks/ms * dt
    wu = np.sqrt(-1*(F[2][1]/dt  + F[2][3]/dt)) # should be: -1 * (ks/mu + -(ks + ku)/mu) = -1 * (-ku/mu) = ku/mu
    xi = F[0][2]/dt/(2 * ws) # cs/ms * dt / dt / (2 * ws) = cs/ms/(2*ws) = cs/(ms * 2 * ws)
    return eps, ws, wu, xi, Q, kf_em_params[4], kf_em_params[5], kf_em_params[6], kf_em_params[7]


def estimate_model_parameters(true_car, true_road_sigma, acc_noise_sigma, velocity, measured_accs, dt, F_known, Q_known, 
fu_mu_known, n_param_inits, order, x0, P0):
    """
    This function provides the steps for estimating any unkown parameters using the EM algorithm for the kalman filter under a QC model.
    :param true_car: A `QC` instance that represents the information of the car that drove over the profile
    :param true_road_sigma: The "true" value of the variance of the road profile in the time domain.
    :param acc_noise_sigma: The value of the variance of the accelerometer measurement noise.
    :param velocity: The value of the velocity (in units m/s) of the car as it drove over the profile.
    :param measured_accs: The time series of vertical acceleration measurements of the sprung mass (in units m/s**2)
    :param dt: The spacing in the time domain of the time series.
    :param F_known: A flag (assumed to be either 0 or 1) which denotes whether F will be known when running the kalman filter.
    :param Q_known: A flag (assumed to be either 0 or 1) which denotes whether Q will be known when running the kalman filter.
    :param fu_mu_known: A flag (assumed to be either a 0 or 1) which denots whether ku/mu will be known when running the kalman filter.
    :param n_param_inits: An int representing the number of random parameter initializations to run the EM algorithm with.
    :param order: The order of the taylor series expansion to use for the state space matrix.
    :param x0: An ndarray representing the value of the initial state. Should be nx1 (n = # of states, in this case should be 4).
    :param P0: An ndarray representing the covariance of the initial state, should be nxn ndarray.
    :return: ret_dict_avg (Dictionary with the values of the estimated ML parameters averaged over the n_param_inits simulations),
            ret_dict_max (Dictionary with the values of the estimated ML parameters of the iteration with the highest log likelihood),
            ret_dict_var (Dictionary with the values of the variance of the estimated ML parameters over the n_param_inits iterations)
    """
    # Function for producing H if updating F
    def h_func(F, dt):
        H_new = (F[0] + np.array([-1, 0, 0, 0]))/dt
        return  H_new.reshape(1, -1)
    #Note: Since not taking in times we are assuming that there are no missing values

    eps_list, ws_list, wu_list, xi_list, Q_list, x0_list, P0_list, x_list, P_list = [], [], [], [], [], [], [], [], [] # store the values of the estimated parameters over the n_param init initializations
    eps_maxll, ws_maxll, wu_maxll, xi_maxll, Q_maxll, x0_maxll, P0_maxll, x_maxll, P_maxll = None, None, None, None, None, None, None, None, None # store max likelihood values of estimated params

    max_ll = None # Store value of highest likelihood
    for _ in range(0, n_param_inits):
        #Need samples of all the other ones as well?
        #x0 = np.array([[0], [0], [0], [0]])
        #P0 = np.eye(4) * p
        if fu_mu_known: # If you know the value of ku/mu - Idk why I did fu_mu everywhere...
            F, H, Q, R, M, G, params_to_optimize = initialize_kf_est_params(true_car, true_road_sigma, F_known, Q_known, None, acc_noise_sigma, dt, order)
            param_list = [[x0, P0, measured_accs, F, H, Q, R, M, G, dt, params_to_optimize, h_func]] # just one set of params to try
        else:
            for ku_mu in np.arange(40, 400, 10): # Need to optimize over ku_mu...
                F, H, Q, R, M, G, params_to_optimize = initialize_kf_est_params(true_car, true_road_sigma,  F_known, Q_known, ku_mu**2, acc_noise_sigma, dt, order)
                param_list.append([x0, P0, measured_accs, F, H, Q, R, M, G, dt, params_to_optimize, h_func])
        
        log_l_max_this = None # log likelihood for this iteration in case we need to optimize over ku_mu
        result_final = None # Final result in case optimizing over ku_mu
        for params in param_list:
            result = qckalman.maximize_likelihood(*params, max_iters = 25, eps=1, to_plot=False, constrained=True)
            log_l = result[-1]
            #F_new, H_new, Q_new, R_new, x0_new, P0_new, x_smoothed_new, P_smoothed_new, log_l = qckalman.maximize_likelihood(*params, max_iters = 50, eps=1, to_plot=False, constrained=True)
            
            if result_final is None or log_l > log_l_max_this: # We are just going to find the max likelihood over ku_mu and return that 
                result_final = result
                log_l_max_this = log_l
                #F_final, H_final, Q_final, R_final, x0_final, P0_final, x_smoothed_final, P_smoothed_final = F_new, F_new, H_new, Q_new, R_new, x0_new, P0_new, x_smoothed_new, P_smoothed_new
        eps, ws, wu, xi, Q, x0_new, P0_new, xs, Ps = reformat_params(result_final, dt) # get our param values
        
        if max_ll is None or log_l_max_this > max_ll: # Set maximum likelihood values 
            eps_maxll, ws_maxll, wu_maxll, xi_maxll, Q_maxll, x0_maxll, P0_maxll, x_maxll, P_maxll = eps, ws, wu, xi, Q, x0_new, P0_new, xs, Ps
            max_ll = log_l_max_this # make sure we store this value for the next iteration

        # Append all values to the lists
        eps_list.append(eps)
        ws_list.append(ws)
        wu_list.append(wu)
        xi_list.append(xi)
        Q_list.append(Q)
        x_list.append(xs)
        P_list.append(Ps)
        x0_list.append(x0_new)
        P0_list.append(P0_new)


        # 2. Append parameters to dictionary
        #add_est_car_params(to_ret_dict, dt, *result_final)
    avg_eps, var_eps = np.mean(eps_list), np.var(eps_list)
    avg_ws, var_ws = np.mean(ws_list), np.var(ws_list)
    avg_wu, var_wu = np.mean(wu_list), np.var(wu_list)
    avg_xi, var_xi = np.mean(xi_list), np.var(xi_list)
    
    avg_Q, var_Q = np.mean(Q_list, axis=0), np.var(Q_list, axis=0)
    avg_xs = np.mean(x_list, axis=0)
    avg_Ps = np.mean(P_list, axis=0)
    avg_x0 = np.mean(x0_list, axis=0)
    avg_P0 = np.mean(P0_list, axis=0)
    ret_dict_avg = {'Est_Eps_Avg': avg_eps, 'Est_Ws_Avg': avg_ws, 
    'Est_Wu_Avg': avg_wu, 'Est_Xi_Avg': avg_xi,  'Est_Road_Sigma_Avg': avg_Q[-1][-1], 
    'Avg_X0': avg_x0, 'Avg_P0': avg_P0, 'Avg_Xs': avg_xs, 'Avg_Ps': avg_Ps} # MAke the return dictionaries
    
    ret_dict_max = {'Est_Eps_Max': eps_maxll, 'Est_Ws_Max': ws_maxll, 
    'Est_Wu_Max': wu_maxll,'Est_Xi_Max': xi_maxll, 'Est_Road_Sigma_Max': Q_maxll[-1][-1], 
    'Max_X0': x0_maxll, 'Max_P0': P0_maxll, 'Max_Xs': x_maxll, 'Max_Ps': P_maxll}

    ret_dict_var = {'Var_Eps': var_eps, 'Var_Ws': var_ws, 'Var_Wu': var_wu, 'Var_Xi': var_xi,'Var_Road_Sigma': var_Q[-1][-1]}
    return ret_dict_avg, ret_dict_max, ret_dict_var


# def estimate_profile(return_dict, F_est, H_est, Q_est, R_est, x0_est, P0_est, x_smoothed_est, P_smoothed_est, log_l, dt, accs, input_profile):
#     ku_mu_est = F_est[2][1] + F_est[2][3]
#     F_prof = np.vstack([np.hstack([F_est, np.array([[0], [0], [ku_mu_est], [0]])]), np.array([[0, 0, 0, 0, 1]])])
#     Q_prof = np.zeros((5, 5))
#     Q_prof[-1][-1] = Q_est[-1][-1]
#     H_prof = np.vstack([np.hstack([H_est, np.array([[0]])]), 
#     np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, -1], [0, 0, 1, 0, 0], [0, 0, 0, 1, -1]])])
#     R_prof = np.zeros((5, 5))
#     R_prof[0][0] = R_est[0][0]
#     R_mean = np.mean(P_smoothed_est, axis=0) #Estimate error covariance matrix by taking mean of estimate error covariances (I know this isn't kosher per say, but might as well try)
#     for x in range(1, 5):
#         R_prof[x][x] = R_mean[x-1][x-1]
#     x0 = np.vstack([x0_est, np.array([[0]])])
#     P0 = np.vstack([np.hstack([P0_est, np.array([[0], [0], [0], [0]])]), np.array([[0, 0, 0, 0, Q_est[0][0]]])])
#     zs = np.transpose(np.vstack([accs, np.hstack(x_smoothed_est)]))
#     xs_for, P_preds_for, P_upds_for, log_l_for, K_for = qckalman.kf_forward(x0, P0, F_prof, H_prof, Q_prof, R_prof, zs)
#     x0_final, P0_final, xs_final, Ps_final, _, _, _, _ = qckalman.kf_backward(x0, P0, F_prof, H_prof, Q_prof, R_prof, zs, xs_for, P_preds_for, P_upds_for, K_for)
#     xs_final_stacked = np.hstack(xs_final)
#     xs_dot_final = xs_final[0]
#     xs_final = xs_final[1]
#     xu_dot_final = xs_final[2]
#     xu_final = xs_final[3]
#     prof_final = xs_final[4]
#     return_dict['Xs_Dot_Est'] = xs_dot_final
#     return_dict['Xs_Est'] = xs_final
#     return_dict['Xu_Dot_Est'] = xu_dot_final
#     return_dict['Xu_Est'] = xu_final
#     return_dict['Prof_Est'] = prof_final
#     return_dict['Input_Prof'] = input_profile
#     #return xs_final, xu_dot_final, xu_final, prof_final, Ps_final


def initialize_kf_est_profile(est_eps, est_ws, est_wu, est_xi, est_road_sigma, est_Ps, acc_sigma, dt):
    """
    This function helps to initialize the input for the kalman filter when you are ready to estimate the actual road profile input. 
    Assumes a 5-dimensional state space of: x_dot_s (sprung mass velocity in units m/s), 
                                            x_s (sprung mass displacement in units m), 
                                            x_dot_u (unsprung mass velocity in units m/s), 
                                            x_u (unsprung mass displacement in units m), 
                                            y (elevation of road profile in units m)
    
    Assumes a 5-dimensional measurment of:  x_dot_dot_s (unsprung mass acceleration in units m/s**2),
                                            x_dot_s (sprung mass velocity in units m/s), 
                                            x_s - y (sprung mass displacement - road profile elevation in units m), 
                                            x_dot_u (unsprung mass velocity in units m/s), 
                                            x_u - y (unsprung mass displacement - road profile elvation in units m)

    :param est_eps: The estimated value of epsilon returned from the first kalman filter output (ms/mu).
    :param est_ws: The estimated value of omega_s returned from the first kalman filter output (sqrt(ks/ms))
    :param est_wu: The estimated value of omega_u returned from the first kalman fitler output (sqrt(ku/mu))
    :param est_xi: The estimated value of xi returned from the first kalman filter output (cs / (2 * ms * ws))
    :param est_road_sigma: The estimated value of the road profile variance in the time domain.
    :param acc_sigma: The value of the acceleration noise variance 
    :param dt: The spacing of the measurements in the time domain.
    :return: F_final (the estimated state transition matrix), H (The estimated measurement matrix), 
            Q (The estimated process noise covariance matrix), R_prof (The estimated measurement noise covariance matrix)
    
    """
    # Just so less typing...
    eps = est_eps
    ws = est_ws
    wu = est_wu
    xi = est_xi
# F = np.array([[-1*cs_ms, -1*ks_ms, cs_ms, ks_ms], 
#         [1, 0, 0, 0], 
#         [cs_mu, ks_mu, -1*cs_mu, -1*ku_ks_mu], 
#         [0 , 0, 1, 0]]) # initialize F
    F = np.array([[-1*xi*2*ws, -1*ws**2, xi*2*ws, ws**2, 0], 
    [1, 0, 0, 0, 0], 
    [xi*2*ws*eps, ws**2*eps, -1*xi*2*ws*eps, -1*(ws**2*eps + wu**2), wu**2], 
    [0, 0, 1, 0, 0], 
    [0, 0, 0, 0, 0]]) # Checking work...
                      # 2 * cs  * ws / (2 * ms * ws) = cs/ms; sqrt(ks/ms)**2 = ks/ms
                      # 2 * cs * ws * ms /( 2 * ms * ws * mu) = cs/mu
                      # sqrt(ks/ms)**2 * ms/mu = ks/ms * ms/mu = ks/mu
                      # -1 * sqrt(ks/ms)**2 * ms/mu * sqrt(ku/mu**2) = -1*(ks/mu + ku/mu)
    F_final = expm(F * dt) # Just go all in on this one

    H = np.array([[-1*xi*2*ws, -1*ws**2, xi*2*ws, ws**2, 0], 
    [1, 0, 0, 0, 0], 
    [0, 1, 0, 0, -1], 
    [0, 0, 1, 0, 0], 
    [0, 0, 0, 1, -1]]) 

    Q = np.array([[0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0], 
        [0, 0, 0, 0, est_road_sigma]])
    
    R_prof = np.zeros((5, 5))
    R_prof[0][0] = acc_sigma
    R_mean = np.mean(est_Ps, axis=0) #Estimate error covariance matrix by taking mean of estimate error covariances (I know this isn't kosher per say, but might as well try)
    R_prof[1:, 1:] = R_mean  #* 1000000
    R_prof = (R_prof + R_prof)/2
    return F_final, H, Q, R_prof
    
    


def estimate_profile(est_eps, est_ws, est_wu, est_xi, est_road_sigma, est_x0, est_P0,  est_xs, est_Ps, sprung_accs, acc_sigma, dt):
    """
    This function defines the steps for estimating the road profile once the kalman filter for estimating the car/noise parameters
    and non-measured states has been run.
    :param est_eps: The estimated value of epsilon returned from the first kalman filter output (ms/mu).
    :param est_ws: The estimated value of omega_s returned from the first kalman filter output (sqrt(ks/ms))
    :param est_wu: The estimated value of omega_u returned from the first kalman fitler output (sqrt(ku/mu))
    :param est_xi: The estimated value of xi returned from the first kalman filter output (cs / (2 * ms * ws))
    :param est_road_sigma: The estimated value of the road profile variance in the time domain. 
    :param est_x0: The estimated value of the initial state.
    :param est_P0: The estimated value of the initial state covariance.
    :param est_xs: The estimated values of the non-measured states produced by the first kalman filter as defined in estimate_model_params().
    :param est_Ps: The estimated covariances of the non-measured states produced by the first kalman filter as defined in estimate_model_params().
    :param sprung_accs: The measured sprung acceleration values (in units m/s**2). Should be of shape Tx1.
    :param acc_sigma: The variance of the acceleration measruement noise.
    :param dt: The spacing of the measurements in the time domain.
    :return: prof_final (The final estimated profile, in units m)
    """
    F_prof, H_prof, Q_prof, R_prof = initialize_kf_est_profile(est_eps, est_ws, est_wu, est_xi, est_road_sigma,est_Ps, acc_sigma, dt)
    x0 = np.vstack([est_x0, np.array([[0]])]) # Use an estimate of 0 for the initial elevation
    P0 = np.vstack([np.hstack([est_P0, np.array([[0], [0], [0], [0]])]), np.array([[0, 0, 0, 0, est_road_sigma]])]) # add covariances of 0 for all states with road profile, use est_road_sigma as variance for initial road profile
    #zs_prof = np.transpose(np.hstack([sprung_accs, np.hstack(est_xs)]))
    zs_prof = np.hstack([sprung_accs, np.transpose(np.hstack(est_xs))]) # Rows should be: [sprung_acc, sprung_vel, sprung_disp - prof, unsprung_vel, unsprung_vel - prof]
    #print("zs_prof.shape is {0}".format(zs_prof.shape))
    #sys.exit(0)
    # Run the kalman filter
    xs_for, P_preds_for, P_upds_for, log_l_for, K_for = qckalman.kf_forward(x0, P0, F_prof, H_prof, Q_prof, R_prof, zs_prof)
    x0_final, P0_final, xs_final, Ps_final, _, _, _, _ = qckalman.kf_backward(x0, P0, F_prof, H_prof, Q_prof, R_prof, zs_prof, xs_for, P_preds_for, P_upds_for, K_for)
    prof_final = np.array(list(map(lambda x: x[4][0], xs_final))) #Finally, extract just the profile estimate
    return prof_final#, zs_prof



    


# def kf_exp_datagenerator(data, sample_rates, Q_known, F_known, fu_mu_known, acc_noise, n_param_inits, order, profile_dict):
#     #df = pd.DataFrame(data)
#     for row in data.itertuples(index=False):
#         temp_dict = row._asdict()
#         temp_dict['sample_rates'] = sample_rates
#         temp_dict['Q_known'] = Q_known
#         temp_dict['F_known'] = F_known
#         temp_dict['fu_mu_known'] = fu_mu_known
#         temp_dict['acc_noises'] = acc_noise
#         temp_dict['n_param_inits'] = n_param_inits
#         temp_dict['order'] = order
#         #temp_dict['seed'] = start_seed
#         yield (temp_dict, profile_dict)



def add_est_params(ret_dict, param_dicts, bad_keys=set(['Avg_Xs', 'Avg_Ps', 'Max_Xs', 'Max_Ps', 'Avg_X0', 'Avg_P0', 'Max_X0', 'Max_P0'])):
    """
    Another handy helper function for adding data to the dictionary we will be returning. Assumes all keys in param_dicts
    are also in ret_dict, aside from the keys in bad_keys (which should be some form of iterable, ideally a set).
    :param ret_dict: The dictionary we'll be returning at the very end, contains all the data from the experiment.
    :param param_dicts: The output from the estimation of the parameters from the EM kalman filter.
    :param bad_keys: Keys from the param_dicts that are not in ret_dict
    """
    for param_dict in param_dicts:
        for k in param_dict:
            if k not in bad_keys:
                ret_dict[k].append(param_dict[k])

def add_profile_error_stats(ret_dict, est_elevations, true_profile, sampled_profile, est_type):
    """
    Yep, that's right, another helper function to add things to the data dictionary! This also does some other work,
    such as computing the MSE and MAE of the estimated profile with the true profile, computing the IRI of the estimated profile,
    and computing the errors for the IRI over each 50 and 100 meter section of the two profiles.
    :param ret_dict: The dictionary holding all the precious experiment data.
    :param sampled_dists: The distances at which the profile was sampled by the car driving over it. 
    :param est_elevations: The estimated elevations as returned by the kalman filter.
    :param true_profile: An instance of `RoadProfile` which contains the original road profile in the spatial domain.
    :param input_profile: An instance of `RoadProfile` which contains the sampled road profile in the time domain, input to the qc simulation at 1000 Hz.
    :param est_type: Should be one of 'Avg' or 'Max'. The type of parameter estimate, either avg over all random initializations or the maximum likelihood estimate.
    """
    #what units to use - let's use mm
    # Create an instance of roadprofile for convenience when computing the IRI's - 
    #total_time = len(est_elevations)/sample_rate_hz  # total_time = total_samples/samples_per_second
    #total_distance = total_time * velocity # d/t = v, therefore d = t * v
    #est_dists = np.linspace(0, total_distance, len(est_elevations)) # evenly space the measurements
    est_profile = roadprofile.RoadProfile(distances=sampled_profile.get_distances(), elevations=est_elevations*1000)
    st_index = np.where(est_profile.get_distances() >= 50)[0][0] # Ignore first 50 meters due to error of initial conditions

    mse = np.mean((est_profile.get_elevations()[st_index:] - sampled_profile.get_elevations()[st_index:])**2)
    mae = np.mean(np.abs(est_profile.get_elevations()[st_index:] - sampled_profile.get_elevations()[st_index:]))
    ret_dict['Prof_MSE_{0}'.format(est_type)].append(mse)
    ret_dict['Prof_MAE_{0}'.format(est_type)].append(mae)
    iris_50_true = [] # Stores iris over each 50m section of the profile
    iris_50_est = []
    iris_100_true = [] # Stores iris over each 100m section of the profile
    iris_100_est = []
    plen = true_profile.get_distances()[-1]
    dx1 = np.diff(true_profile.get_distances()[:2])[0]
    dx2 = np.diff(est_profile.get_distances()[:2])[0]
    tot_segments = int(plen/50)
    step1, step2 = int(50/dx1), int(50/dx2) 
    st_ind1, st_ind2 = step1, step2
    
    
    e_ind1, e_ind2 = 2*step1, 2*step2
    iri_true_prev, iri_est_prev = None, None
    # Compute IRI error over each 50 and 100 meter segments, skipping the first 50 meters
    logging.debug("Len true: {0}, len est: {1}".format(len(true_profile.get_distances()), len(est_profile.get_distances())))
    logging.debug("tot segments is {0}".format(tot_segments))
    for x in range(1, tot_segments):
        logging.debug("it = {0}, st_ind1: {1} st_ind2 = {2}".format(x, st_ind1, st_ind2))
        logging.debug("it = {0}, e_ind1: {1} e_ind2 = {2}".format(x, e_ind1, e_ind2))
        
        iri_true = true_profile.to_iri(st_ind1, e_ind1+1)
        iri_est = est_profile.to_iri(st_ind2, e_ind2+1)
        st_ind1, st_ind2 = e_ind1, e_ind2
        e_ind1 += step1
        e_ind2 += step2
        iris_50_true.append(iri_true)
        iris_50_est.append(iri_est)
        if iri_true_prev is not None:
            iris_100_true.append((iri_true + iri_true_prev)/2)
            iris_100_est.append((iri_est + iri_est_prev)/2)
        iri_true_prev = iri_true
        iri_est_prev = iri_est
    
    # Now, compute the IRI errors
    errors_50 = np.abs(np.array(iris_50_true) - np.array(iris_50_est))
    errors_100 = np.abs(np.array(iris_100_true) - np.array(iris_100_est))
    mse_50 = np.mean(errors_50**2)
    mae_50 = np.mean(errors_50)
    mse_100 = np.mean(errors_100**2)
    mae_100 = np.mean(errors_100)
    min_50, max_50 = np.min(errors_50), np.max(errors_50)
    min_100, max_100 = np.min(errors_100), np.max(errors_100)
    ret_dict['IRI_{0}'.format(est_type)].append(est_profile.to_iri())
    ret_dict['MSE_50_{0}'.format(est_type)].append(mse_50)
    ret_dict['MAE_50_{0}'.format(est_type)].append(mae_50)
    ret_dict['MSE_100_{0}'.format(est_type)].append(mse_100)
    ret_dict['MAE_100_{0}'.format(est_type)].append(mae_100)
    ret_dict['Max_E_50_{0}'.format(est_type)].append(max_50)
    ret_dict['Min_E_50_{0}'.format(est_type)].append(min_50)
    ret_dict['Max_E_100_{0}'.format(est_type)].append(max_100)
    ret_dict['Min_E_100_{0}'.format(est_type)].append(min_100)


    
    

def estimate(road_type, road_length, road_number, profile, velocities, gn0, output_directory, acc_noise,
                F_known, Q_known, fu_mu_known, sample_rates, n_param_inits, order, orig_sr_hz=1000, P0s=[.001, .01, .1]):
    """
    Function that defines all the steps for:
    1. Randomly picking a car to drive over a road profile.
    2. Simulating that car driving over the profile at different speeds.
    3. Estimating the car parameters and sprung/unsprung velocities and displacements just from the sampled vertical acceleration series of its sprung mass.
    4. Estimating the profile it drove over from step 3.
    5. Computing the error of the estimates of 3 and 4 and saving that to a file, so we can see how well this method does.

    :param road_type: The class of road, should be one of A, B, C, D, E, F, G, or H. Only used for saving file. 
    :param road_length: The length of the road in units m. Only used for file name.
    :param road_number: The number of the road. Only used for file name.
    :param profile: The `RoadProfile` instance which represents the road that the car will be driving on.
    :param velocities: A list of integer velocities (in units m/s) that contains the speeds over which to drive over the profile.
    :param gn0: The "roughness" coefficient of the road profile.
    :param output_directory: A path to the directory where the data should be saved.
    :param acc_noise: A list representing the variances of the acceleration series noise.
    :param F_known: A flag (should be 0/1 or True/False) representing whether or not F should be known for this simulation.
    :param Q_known: A flag (should be 0/1 or True/False) representing whether or not Q should be known for this simulation.
    :param fu_mu_known: A flag (should be 0/1 or True/False) representing whether or not ku/mu should be known for this simulation.
    :param sample_rates: A list of integer sampling rates that represent at what frequency in the time domain that the acceleration series 
                            should be sampled. Units are Hz.
    :param n_param_inits: An integer representing the number of random parameter initializations to use for the estimation.
    :param order: The order of Taylor series expansion to use for the state transition matrix.
    :param orig_sr_hz: The sampling rate (in units Hz) to use for the original sampling of the acceleration series. Default: 1000
    :param P0s: A list of the initial covariances to use for the initial state. Default: [.001, .01, .1]
    :return: The runtime in seconds of the entire simulation, the road_type, the road_length, and the road_number 
    (the return values are just to help with logging)

    """
    # What our output data will look like:
    # road_type, road_length, road_number, gn0, car, velocity, sample_rate, acc_noise, 
    # true_ks, true_ku, true_ms, true_mu, 
    


    # All the things that we are going to compute - kind of annoying, but couldn't think of a better design.
    ret_dict = {'Gn0': [], 'Car': [], 'Velocity': [], 'Sample_Rate': [], 'Acc_Noise': [], 
    'P0': [], 'True_Road_Sigma': [], 'Est_Road_Sigma_Avg': [], 'Var_Road_Sigma': [], 'Est_Road_Sigma_Max': [], 'True_Eps': [], 'True_Ws': [], 'True_Wu': [], 'True_Xi': [], 
    'Est_Eps_Avg': [], 'Est_Ws_Avg': [], 'Est_Wu_Avg': [], 'Est_Xi_Avg':[], 'Var_Eps': [], 'Var_Ws': [], 'Var_Wu': [], 
    'Var_Xi': [], 'Est_Eps_Max': [], 'Est_Ws_Max': [], 'Est_Wu_Max': [], 'Est_Xi_Max': [], 
    'Prof_MSE_Avg': [], 'Prof_MAE_Avg':[], 'Prof_MSE_Max': [], 'Prof_MAE_Max':[],'IRI_True': [], 'IRI_Avg': [], 'IRI_Max': [], 
    'MSE_50_Avg': [], 'MAE_50_Avg': [], 'MSE_100_Avg': [], 'MAE_100_Avg': [],'Min_E_50_Avg': [], 'Max_E_50_Avg': [], 'Min_E_100_Avg': [], 'Max_E_100_Avg': [], 
    'MSE_50_Max': [], 'MAE_50_Max': [], 'MSE_100_Max': [], 'MAE_100_Max': [],'Min_E_50_Max': [], 'Max_E_50_Max': [], 'Min_E_100_Max': [], 'Max_E_100_Max': []}
    c_list = cars.get_car_list() # Get the list of cars
    st_time = time.time() # get the start time.
    for v in velocities:
        
        #for car_tup in cars.get_car_list(): #This way, does acceleration for each car only once
        car_ind = np.random.choice(len(c_list)) # randomly pick a car to use for each velocity
        car_tup = c_list[car_ind]
        cname, true_car = car_tup[0], car_tup[1]
        true_iri = profile.to_iri()
        # Generate the series of sprung accelerations
        T, yout, xout, new_dists, new_els, vs = true_car.run2(profile, [], v, final_sample_rate=orig_sr_hz) #always sample at 1000 Hz, the max rate
        #input_profile = roadprofile.RoadProfile(distances=new_dists, elevations=new_els)
        sprung_accs = yout[:, -1] # Probably should change that function...
        #avg_params, max_l_params = estimate_car_parameters(ret_dict, gn0, profile, cname, car, v, sprung_acces, new_els, 
        #sample_rates, F_known, Q_known, fu_mu_known, acc_noise, n_param_inits, order)
        for sr_hz in sample_rates:
            dt = 1/sr_hz # space between samples, reciprocal of sampling rate.
            factor = int(orig_sr_hz/sr_hz) # the downsampling rate
            # Downsample
            sampled_dists = new_dists[list(range(0, len(new_dists), factor))]
            sampled_accs = sprung_accs[list(range(0, len(sprung_accs), factor))]
            
            sampled_els = new_els[list(range(0, len(new_els), factor))]
            # For convenience, create a RoadProfile instance of the sampled profile
            sampled_profile = roadprofile.RoadProfile(distances=sampled_dists, elevations=sampled_els)
            true_road_sigma = np.var(np.diff(sampled_els/1000)) # calculate the variance of the profile in the time domain - need to divide by 1000 since originally is in mm
            zs = sampled_accs.reshape(-1, 1) # reshape the sampled accelerations so that each row contains one sample
            for sigma in acc_noise:
                added_noise = np.random.normal(loc=0, scale=np.sqrt(sigma), size=zs.shape) # Generate the noise
                accs_plus_noise = zs + added_noise # add the noise
                for p in P0s:
                    x0 = np.array([[0], [0], [0], [0]])
                    P0 = np.eye(4) * p # initialize P0 as diagonal (so each component initially independent)
                    ret_dict['Gn0'].append(gn0) # append gn0
                    #ret_dict['Car'].append(cname) # just going to do this in the add_true_car_params helper function
                    ret_dict['Velocity'].append(v) # append the velocity
                    ret_dict['Sample_Rate'].append(sr_hz) # append sampling rate
                    ret_dict['Acc_Noise'].append(sigma) # append the noise level
                    ret_dict['P0'].append(p) # append initial covariance
                    ret_dict['True_Road_Sigma'].append(true_road_sigma) # append the true process noise variance
                    ret_dict['IRI_True'].append(true_iri)
                    add_true_car_params(ret_dict, cname, true_car) # Append the car parameters

                    # Now, estimate whichever parameters we're supposed to! 
                    est_param_dicts = estimate_model_parameters(true_car, true_road_sigma, sigma, v, accs_plus_noise, dt, F_known, Q_known, 
                        fu_mu_known, n_param_inits, order, x0, P0)

                    # Add these estiamted parameters
                    add_est_params(ret_dict, est_param_dicts)
                    # Now, estimate the profile using the two estimates

                    profile_est_avg = estimate_profile(*est_param_dicts[0].values(), accs_plus_noise, sigma, dt)
                    profile_est_max = estimate_profile(*est_param_dicts[1].values(), accs_plus_noise, sigma, dt)
                    
                    # Compute the errors and add them
                    add_profile_error_stats(ret_dict, profile_est_avg, profile, sampled_profile, 'Avg')
                    add_profile_error_stats(ret_dict, profile_est_max, profile, sampled_profile, 'Max')
    # Finally, save the file
    try:
        with gzip.open("{0}{1}_{2}_{3}.pickle.gz".format(output_directory, road_type, road_length, road_number), 'wb') as f:
            pickle.dump(ret_dict, f)
    except Exception as e:
        logging.error("Failed saving file for {0}_{1}_{2}, exception was {3}".format(road_type, road_length, road_number, e))
    #car_param_info = estimate_model_parameters(ret_dict, profile, args['car'], true_car, *args[0].values()[5:])
    #estimate_profile(ret_dict, *car_param_info[0], car_param_info[1], car_param_info[2], car_param_info[3])
    return time.time() - st_time, road_type, road_length, road_number #or save to file?

def estimate_star(args):
    """
    This function is a dummy function - all it does is call the real estimate function, since by default python's 
    process pool can only take in one argument, not a list of them.
    """
    # If only it was actually that easy...
    return estimate(**args)

# def get_profile_dict(profile_directory):
#     curr_dists = None
#     curr_els = None
#     prof_dict = {}
#     for fname in sorted(os.listdir(profile_directory)):
#         fsplit = fname[:-4].strip().split('_')
#         if fsplit[-1] == 'distances':
#             if curr_dists is None:
#                 curr_dists = np.load(profile_directory + fname)
#             else:
#                 raise ValueError("Obtained two distances in a row, curr fname is {0}".format(fname))
#         elif fsplit[-1] == 'elevations':
#             if curr_els is None:
#                 curr_els = np.load(profile_directory + fname)
#             else:
#                 raise ValueError("Obtained two elevations in a row, curr fname is {0}".format(fname))

#         if curr_dists is not None and curr_els is not None:
#             profile = roadprofile.RoadProfile(distances=curr_dists, elevations=curr_els)
#             prof_dict['_'.join(fsplit[1:4])] = profile
#             curr_dists, curr_els = None, None
#         #Profile_C_5000_29_distances.npy
#     return prof_dict


# def start_processing_accs(input_file, output_file, profile_directory, num_profiles, sample_rates, Q_known, F_known, 
#     fu_mu_known, acc_noise, n_param_inits, order, start_seed):
#     #with gzip.open(input_file, 'rb') as f:
#     #    data_dict = pd.DataFrame(pickle.load(f)) # This should be the acceleration data - instead, we are going to generate it during this program
#     profile_dict = get_profile_dict(profile_directory)
#     #data_iterator = kf_exp_datagenerator(data_dict, sample_rates, Q_known, F_known,
#     #fu_mu_known, acc_noise, n_param_inits, order, profile_dict)
#     data_iterator = kf_exp_datagenerator(data_dict, sample_rates, Q_known, F_known,
#     fu_mu_known, acc_noise, n_param_inits, order, profile_directory)
#     total_to_process = n_param_inits * len(sample_rates) * len(acc_noise) * 3 * (0 if fu_mu_known else 1584) * 8 * num_profiles * 10 * 90 * 5
#     chunk_s = 64
#     logging.info("Starting process pool")
#     with Pool(1) as pool:
#         final_list = []
#         kf_result_unordered_map = pool.imap_unordered(estimate_star, data_iterator, chunksize=chunk_s) #Idk
#         st_time = time.time()
#         logging.info("Entering process loop")
#         for element in kf_result_unordered_map:
            
#             final_list.append(element)
#             logging.info("Processed {0} in {1} seconds, only {2} remaining".format(chunk_s, time.time() - st_time, total_to_process))
#             total_to_process -= chunk_s


#     with gzip.open(output_file, 'wb') as out_f:
#         pickle.dump(final_list, out_f)
    
        
        #Add a process to the queue?


    
#def parse_args_and_run():

    
#parse_args_and_run()



