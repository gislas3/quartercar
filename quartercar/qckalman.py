import numpy as np
from matplotlib import pyplot as plt
#from filterpy.kalman import KalmanFilter
#from filterpy.common import Q_continuous_white_noise
#from experiments import computeisoroughness as cisr
#from quartercar import roadprofile as rp
#from quartercar import qc
#from tests import make_profile as mp
#from scipy.signal import periodogram, welch, detrend, butter, sosfilt
#from scipy.integrate import simps, cumtrapz
#from scipy.stats import norm as st_norm
#from scipy.stats import probplot
#from scipy.linalg import expm
#from math import factorial
import sys

# This program defines the steps for performing kalman filtering, should work generally but it is specifically made right now just to work with the 
# Quartercar state space. It is a TODO to make this into its own separate class/library such that it can be imported ad hoc. Also, would be nice to 
# be able to place this in the same place as the extended kalman filter code for the orientation. Also, it might potentially
# benefit to be able to have this as its own class/Object. That is also a TODO


def kf_forward(x0, P0, F, H, Q, R, zs):
    """
    kf_forward defines the forward kalman filtering sequence for producing the kalman estimates. Right now this only supports
    constant state transition, process noise, measurement, and measurement noise matrices.
    :param x0: The initial state at which to provide the kalman filtering. Should be a 1xn numpy ndarray (i.e. column vector), where n is the dimension of the state vector.
    :param P0: The initial covariance of the initial state estimate, should be an nxn numpy ndarray
    :param F: The state transition matrix of the kalman filter. Should be nxn.
    :param H: The measurement matrix of the kalman filter, which should be a mxn numpy ndarray, m being the dimension of the measurement vector.
    :param Q: The process noise covariance, should be a nxn numpy ndarray.
    :param R: The measurement noise, should be a mxm numpy ndarray.
    :param zs: The measurements that correspond with the state at each point in time - should be a Txm ndarray, i.e. each row of the array represents a measurement.
    :return: xs (the sequence of state estimates produced by the filter), P_preds (the sequence of estimate covariance matrix priors), 
    P_upds (the sequence of the estimate covariance matrix posteriors), log_l (the log likelihood of the state sequence under the model), K (the final kalman gain, needed for the kalman smoother)
    """
    xs = [] # store states in a list
    P_preds, P_upds = [], [] # store estimate covariances in a list as well
    log_l = 0
    x_curr = x0 # initialize states
    P_curr = P0
    F_t = np.transpose(F) # handy so don't have to do this in the loop each time - as mentioned above, this assumes that F is constant for all input times
    H_t = np.transpose(H)

    for idx in range(0, len(zs)):
        y = zs[idx].reshape(R.shape[0], -1) # reshape to column vector

        x_pred = np.matmul(F, x_curr) # Get the prior prediction

        P_pred = np.linalg.multi_dot([F, P_curr, F_t]) + Q # Get the prior covariance

        P_preds.append(P_pred)
        S = np.linalg.multi_dot([H, P_pred, H_t]) + R 

        S_inv = np.linalg.inv(S)
        K = np.linalg.multi_dot([P_pred, H_t, S_inv]) # calculate the kalman gain
        resid = y - np.matmul(H, x_pred) # calculate the residual

        x_upd = x_pred + np.matmul(K, resid) # finally, the posterior prediction
        P_upd = P_pred - np.linalg.multi_dot([K, H, P_pred]) # posterior covariance
        P_upds.append(P_upd)
        xs.append(x_upd)
        x_curr = x_upd #reset the state
        P_curr = P_upd
        term2 = np.linalg.multi_dot([np.transpose(resid), S_inv, resid])

        log_l += -.5 * np.log(np.linalg.det(S)) - .5 * term2[0][0] # compute the log likelihood
    
    return xs, P_preds, P_upds, log_l, K

def kf_backward(x0, P0, F, H, Q, R, zs, x_orig, P_preds, P_upds, K):
    """
    kf_backward computes the smoothed kalman estimates using backwards recursion. It requires that you have produced
    forward estimates of the states of your process using the kf_forward algorithm defined above. Similar to the kf_forward
    algorithm, it assumes that all matrices are constant throughout all the input times. I used the smoothing algorithm as defined
    by Shumway and Stoffer: https://www.stat.pitt.edu/stoffer/dss_files/em.pdf
    :param x0: The initial state at which to provide the kalman filtering. Should be a 1xn numpy ndarray (i.e. column vector), where n is the dimension of the state vector.
    :param P0: The initial covariance of the initial state estimate, should be an nxn numpy ndarray
    :param F: The state transition matrix of the kalman filter. Should be nxn.
    :param H: The measurement matrix of the kalman filter, which should be a mxn numpy ndarray, m being the dimension of the measurement vector.
    :param Q: The process noise covariance, should be a nxn numpy ndarray.
    :param R: The measurement noise, should be a mxm numpy ndarray.
    :param zs: The measurements that correspond with the state at each point in time - should be a Txm ndarray, i.e. each row of the array represents a measurement.
    :param x_orig: A list containing the set of state estimates produced by a forward pass of the kalman filter, so should be length T, where
    each element of the list is a nx1 ndarray.
    :param P_preds: A list containing the set of prior state covariances produced by a forward pass of the kalman filter. So should be 
    length T, and each elment of the list should be a nxn ndarray.
    :param P_upds: A list containing the set of posterior state covariances produced by a forward pass of the kalman filter. So should be
    length T, and each element of the list should be a nxn ndarray.
    :param K: The final gain produced by the kalman filter at the final state. 
    
    :return: x0_new (the smoothed estimate of the initial state), P0_new (the smoothed estimate of the initial covariance), 
    xs_smoothed (the smoothed estimate of the states at each time), Ps_smoothed (the smoothed estimates of the posterior state covariance matrix at each point in time),
    A (matrix used in the EM algorithm), B (matrix used in the EM algorithm), C (matrix used in the EM algorithm), D (matrix used in the EM algorithm)
    """
    F_t = np.transpose(F)
    xs_smoothed = [x_orig[-1]] # store the final estimates, these won't change since they already use all the data
    Ps_smoothed = [P_upds[-1]]
    J_prev = None

    # I believe this is supposed to be the covariance of consecutive states...
    P_t_1_t_2 = np.linalg.multi_dot([np.eye(len(K)) - np.matmul(K, H), F, P_upds[-2]]) # Initialize given last gain
   
    A = np.zeros(P0.shape) #initialize our matrices that we will be returning
    B = np.zeros(P0.shape)
    C = np.zeros(P0.shape)
    D = np.zeros(R.shape)
    P_upds.insert(0, P0) # insert P0 and x0 just for convenience
    x_orig.insert(0, x0)

    for idx in range(len(P_upds) - 2, -1, -1): #loop backwards, see paper linked above for formula
        y = zs[idx].reshape(R.shape[0], -1)
        P_t_1_t_1 = P_upds[idx]
        P_t_t_1 = P_preds[idx]
        x_next = xs_smoothed[0] # represents next state that has already been smoothed
        P_next = Ps_smoothed[0]
        x_prev_upd = x_orig[idx]
        J = np.linalg.multi_dot([P_t_1_t_1, F_t, np.linalg.inv(P_t_t_1)])

        # I don't think I can provide an explanation for each step, so it would probably be better to read the paper and see if you understand/if I made a mistake...
        x_smoothed = x_prev_upd + np.matmul(J, x_next - np.matmul(F, x_prev_upd))
        
        P_smoothed = P_t_1_t_1 + np.linalg.multi_dot([J, P_next - P_t_t_1, np.transpose(J)])

        if J_prev is not None:
            P_t_t = P_upds[idx + 1]
            P_t_1_t_2 = np.matmul(P_t_t, np.transpose(J)) + np.linalg.multi_dot([J_prev, P_t_1_t_2 - np.matmul(F, P_t_t), np.transpose(J)])

        A +=  P_smoothed + np.matmul(x_smoothed, np.transpose(x_smoothed))
        B += P_t_1_t_2 + np.matmul(x_next, np.transpose(x_smoothed))
        C += P_next + np.matmul(x_next, np.transpose(x_next))
        D += np.matmul(y - np.matmul(H, x_next), np.transpose(y - np.matmul(H, x_next))) + np.linalg.multi_dot([H, P_next, np.transpose(H)])
        
  
        xs_smoothed.insert(0, x_smoothed)
        Ps_smoothed.insert(0, P_smoothed)
        J_prev = np.copy(J)

    return xs_smoothed[0], Ps_smoothed[0], xs_smoothed[1:], Ps_smoothed[1:], A, B, C, D


def update(F, Q, A, B, C, M, G, D, n, constrained):
    """
    update produces new estimates of parameter matrices (possibly with constraints). Basically this is the "maximization" step of the
    Expectation-Maximization algorithm.
    :param F: The state transition matrix of the kalman filter. Should be nxn.
    :param Q: The process noise covariance, should be a nxn numpy ndarray.
    :param A: Matrix A as defined in Shumway and Stoffer. 
    :param B: Matrix B as defined in Shumway and Stoffer. 
    :param C: Matrix C as defined in Shumway and Stoffer.
    :param M: Constraint matrix such that M*F = G
    :param G: Constraint matrix such that M*F = G
    :param D: Matrix D as defined in Shumway and Stoffer.
    :param n: The length of the measurement series. 
    :param constrained: Boolean flag representing whether or not F should be constrained.
    :return: F_new (ML update of parameter F), Q_new (ML update of parameter Q), R_new  (ML update of parameter R) 
    """
    M_t = np.transpose(M)
    A_inv = np.linalg.inv(A)
    try:
        M_t_A_M = np.linalg.inv(np.linalg.multi_dot([M_t, A_inv, M]))
        #print("M_t_A_M matrix is {0}".format(np.linalg.multi_dot([M_t, A_inv, M])))

    except Exception:
        #print("Got singular matrix for M_t_A_M, matrix was {0}".format(np.linalg.multi_dot([M_t, A_inv, M])))
        sys.exit(1)

    F_new = np.matmul(B, A_inv)

    Q_new = 1/n * (C - np.linalg.multi_dot([B, A_inv, np.transpose(B)]))
    
    if constrained:
        F_M = np.matmul(F_new, M)
        term1 = np.matmul(F_M - G, M_t_A_M)
        F_new = F_new - np.linalg.multi_dot([term1, M_t, A_inv])

        #Q_new = Q_new + np.matmul(term1, np.transpose(F_M - G))
    Q_new = (np.transpose(Q_new) + Q_new)/2 #for numerical stability
    R_new = 1/n * D
    #h_new_r1 = (F_new[0] + np.array([-1, 0, 0, 0]))/dt
    #H_new = h_new_r1.reshape(1, -1)

    return F_new, Q_new, R_new#, H_new 


def maximize_likelihood(x0, P0, zs, F, H, Q, R, M, G, dt, params_to_optimize, h_func=None, max_iters = 50, eps=1e-1, to_plot=False, constrained=True):
    """
    maximize_likelihood() defines the order of operations for providing the maximum likelihood (using the Expectation-Maximization algorithm) estimate of the
    parameter matrices requested by the user, under a kalman state space represented by F, H, Q, R, and with measurements zs.
    :param x0: The initial state at which to provide the kalman filtering. Should be a 1xn numpy ndarray (i.e. column vector), where n is the dimension of the state vector.
    :param P0: The initial covariance of the initial state estimate, should be an nxn numpy ndarray
    :param zs: The measurements that correspond with the state at each point in time - should be a Txm ndarray, i.e. each row of the array represents a measurement.
    :param F: The state transition matrix of the kalman filter. Should be nxn.
    :param H: The measurement matrix of the kalman filter, which should be a mxn numpy ndarray, m being the dimension of the measurement vector.
    :param Q: The process noise covariance, should be a nxn numpy ndarray.
    :param R: The measurement noise, should be a mxm numpy ndarray.
    :param M: Constraint matrix such that M*F = G
    :param G: Constraint matrix such that M*F = G
    :param dt: The spacing (in units seconds) between measurements. As you can see, it assumes that the spacing is equal for all measurements
    :param params_to_optimize: A list of parameter names over which to optimize the likelihood function. Can be one or more of ['F', 'Q', 'R', 'x0', 'P0'].
    If you don't put any, it will optimize over all the parameters. Parameters not in that list will be ignored.
    :param h_func: User defined function for updating H if it depends on the value of F.
    :param max_iters: The max iterations to perform the likelihood. Default: 50.
    :param eps: The minimum percentage increase at which lower increases between subsequent operations will stop the EM algorithm. E.g. stop when (log_l_n+1 - log_l_n)/100 < eps.
    :param to_plot: A boolean flag indicating whether or not to plot the log_likelihood function after each iteration of the EM algorithm. Default: False
    :param constrained: A boolean flag indicating whether or not to constrain F in the update portion of the algorithm. If set to True, M and G should be specified. Default: True
    :return: F (possible ML estimate of F or just original F), H (possible ML estimate of H or just original H), Q (possible ML estimate of Q or just original Q), 
    R (possible ML estimate of R or just original R), x0 (possible ML estimate of x0 or just original x0), P0 (possible ML estimate of P0 or just original P0), 
    x_smoothed (smoothed estimates of states evaluated at ML estimated params), P_smoothed (smoothed estimates of covariances evaluated at ML params), 
    log_l (final log likelihood) 
    """
    log_ls = [] #keep track of the likelihoods for each iteration
    curr_it = 0 #keep track of iterations
    curr_l = None #keep track of current likelihood

    while True:
        xs, P_preds, P_upds, log_l, K_final = kf_forward(x0, P0, F, H, Q, R, zs) # run forward kalman filter

        #if len(log_ls) > 0:
            #print("Prev ll: {0}, curr ll: {1}".format(log_ls[-1], log_l))
                    
        log_ls.append(log_l)
        #log_ls2.append(log_l2)
        if (curr_l is not None and abs(log_l - curr_l)/100 <= eps) or curr_it >= max_iters: #percentage change
            break
        

        x0_new, P0_new, x_smoothed, P_smoothed, A, B, C, D = kf_backward(x0, P0, F, H, Q, R, zs, xs, P_preds, P_upds, K_final) # backwards pass of algorithm

        F_new, Q_new, R_new = update(F, Q, A, B, C, M, G, D, len(x_smoothed), constrained) # get new parameters
        for param in params_to_optimize: # figure out which ones to optimize
            if param == 'F':
                F = F_new
                H = h_func(F, dt)
            if param == 'Q':
                Q = Q_new
            if param == 'R':
                R = R_new
            if param == 'x0':
                x0 = x0_new
            if param == 'P0':
                P0 = P0_new
            


        curr_l = log_l
        curr_it += 1

    if to_plot: #Plot if needed
        plt.plot(range(1, len(log_ls) + 1), log_ls)
        plt.title("Log Likelihood for {0} iterations".format(len(log_ls)))
        plt.show()

    return F, H, Q, R, x0, P0, x_smoothed, P_smoothed, log_l
