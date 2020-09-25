import numpy as np
from matplotlib import pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_continuous_white_noise
from experiments import computeisoroughness as cisr
from quartercar import roadprofile as rp
from quartercar import qc
from tests import make_profile as mp
from scipy.signal import periodogram, welch, detrend, butter, sosfilt
from scipy.integrate import simps, cumtrapz
from scipy.stats import norm as st_norm
from scipy.stats import probplot
from scipy.linalg import expm
from math import factorial
import sys

#Author's note: Think of this program as a "rough draft" for the kalman filtering algorithims/programs I made later....

def generate_road_profile(profile_len, seg_lens, prof_types, dx=.01, seed=1, use_saved=True):
    #num_segs = 1
    #assume seglens and profile len add up to each other
    if not np.isclose(np.sum(seg_lens), profile_len):
        raise ValueError("Please enter segment lengths that add up to the profile length that you want")
    #seed = 1
    if not use_saved:
        gn0s = []
        dlist = []
        elist = []
        curr_plen = 0
        for seg_len in seg_lens:
            ptype = np.random.choice(prof_types)
            distances, elevations, true_gn0 = mp.make_profile_from_psd(ptype, 'sine', dx, seg_len, seed=seed, ret_gn0=True)
            gn0s.append(true_gn0)
            if len(dlist) != 0:
                distances = distances + dlist[-1][-1] + dx
                elevations[0] = (elevations[1] + elist[-1][-1])/2
                distances = distances[:-1]
                elevations = elevations[:-1]
                dlist.append(distances)
                elist.append(elevations)
                curr_plen += seg_len
            dlist.append(distances)
            elist.append(elevations)
            curr_plen += seg_len
            seed += 1
        distances_final = np.concatenate(dlist)
        elevations_final = np.concatenate(elist)
        #np.save('/git_environment/quartercar/experiments/data/kalman_dists', distances_final)
        #np.save('/git_environment/quartercar/experiments/kalman_els', elevations_final)
        #print("Saved distances and elevations successfully")
    else:
        gn0s = [13.927682145779794]
        distances_final = np.load('/git_environment/quartercar/experiments/data/kalman_dists.npy')
        elevations_final = np.load('/git_environment/quartercar/experiments/data/kalman_els.npy')
    profile = rp.RoadProfile(distances=distances_final, elevations=elevations_final)
    #plt.plot(profile.get_distances(), profile.get_elevations())
    #plt.title("Road Profile")
    #plt.show()
    return profile, gn0s

def drive_over_profile(vehicle, profile, v0, sample_rate_hz, acc_times=[]):
    T, yout, xout, new_dists, new_els, vs = vehicle.run2(profile, acc_times, v0, final_sample_rate=sample_rate_hz)
    return T, yout[:, -1], new_els, xout[:, 1], xout[:, 0], xout[:, 3], xout[:, 2]

def kf_forward(x0, P0, F, H, Q, R, times, zs, true_meas=None):
    xs = []
    P_preds, P_upds = [], []
    log_l = 0
    x_curr = x0
    P_curr = P0
    F_t = np.transpose(F)
    H_t = np.transpose(H)
    # plt.plot(zs[:, 0])
    # plt.title("Accs")
    # plt.show()
    # plt.plot(zs[:, 1])
    # plt.title("Vels")
    # plt.show()
    # plt.plot(zs[:, 2])
    # plt.title("Disps")
    # plt.show()
    #accs = []
    # vels = []
    # disps = []
    # us_vels = []
    # us_disps = []
    for idx in range(0, len(zs)):
        y = zs[idx].reshape(R.shape[0], -1)
        #print("y is {0}".format(y))
        #accs.append(y[0][0])
        # vels.append(y[0][0])
        # disps.append(y[1][0])
        # us_vels.append(y[2][0])
        # us_disps.append(y[3][0])
        #print(y.shape)
        #break
        x_pred = np.matmul(F, x_curr)
        #print("X_pred is {0}, y is {1}".format(x_pred, y))
        P_pred = np.linalg.multi_dot([F, P_curr, F_t]) + Q
        #print("P pred is {0}".format(P_pred))
        #print("")
        P_preds.append(P_pred)
        S = np.linalg.multi_dot([H, P_pred, H_t]) + R
        #print("S is {0}".format(S))
        S_inv = np.linalg.inv(S)
        K = np.linalg.multi_dot([P_pred, H_t, S_inv])
        resid = y - np.matmul(H, x_pred)
        #print("resid is {0}\n".format(resid))
        x_upd = x_pred + np.matmul(K, resid)
        P_upd = P_pred - np.linalg.multi_dot([K, H, P_pred])
        P_upds.append(P_upd)
        xs.append(x_upd)
        x_curr = x_upd
        P_curr = P_upd
        term2 = np.linalg.multi_dot([np.transpose(resid), S_inv, resid])
        #print("Term 2 shape: {0}".format(term2.shape))
        #print("Term 1 val: {0}".format(-.5 * np.log(np.linalg.det(S))))
        log_l += -.5 * np.log(np.linalg.det(S)) - .5 * term2[0][0]
    #zs = np.transpose(np.vstack([accs, vels, disps]))
    #plt.plot(accs, label='Accs from y')
    #plt.plot(zs[:, 0], label='Accs from z')
    # plt.legend()
    # plt.title("INSIDE ACCS")
    # plt.show()
    # plt.plot(vels, label='Vels from y')
    # plt.plot(zs[:, 0], label='Vels from z')
    # plt.plot(true_meas[0], label='Vels w/noise')
    # plt.legend()
    # plt.title("INSIDE VELS")
    # plt.show()
    # plt.plot(disps, label='Disps from y')
    # plt.plot(zs[:, 1], label='Disps from z')
    # plt.plot(true_meas[1], label='Disps w/noise')
    # plt.legend()
    # plt.title("INSIDE DISPS")
    # plt.show()
    # plt.plot(us_vels, label='Us Vels from y')
    # plt.plot(zs[:,2], label='Us Vels from z')
    # plt.plot(true_meas[2], label='Us Vels w/noise')
    # plt.legend()
    # plt.title("INSIDE VELS")
    # plt.show()
    # plt.plot(us_disps, label='US Disps from y')
    # plt.plot(zs[:, 3], label='US Disps from z')
    # plt.plot(true_meas[3], label='Us Disps w/noise')
    # plt.legend()
    # plt.title("INSIDE DISPS")
    # plt.show()
    
    return xs, P_preds, P_upds, log_l, K
#xs, P_preds, P_upds, log_l, K_final

def initialize_kf(m_s, m_u, c_s, k_u, k_s, road_sigma, noise_sigma, v, dt, v_fact=40, 
var_v_e=None, var_d_e=None, var_uve = None, var_ude = None, order=1):
    print("m_s: {0}, m_u: {1}, c_s: {2}, k_u: {3}, k_s: {4}".format(m_s, m_u, c_s, k_u, k_s))
    F = np.array([[-1*c_s/m_s, -1*k_s/m_s, c_s/m_s, k_s/m_s, 0], 
    [1, 0, 0, 0, 0], 
    [c_s/m_u, k_s/m_u, -1*c_s/m_u, -1*(k_u + k_s)/m_u , k_u/m_u], 
    [0 , 0, 1, 0, 0], 
    [0, 0, 0, 0, 0]]) * dt
    F = np.eye(len(F)) + F

    Q = np.array([[0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0], 
    [0, 0, dt**3*k_u**2/(3*m_u**2), 0, dt**2*k_u/(2*m_u)],
    [0, 0, 0, 0, 0], 
    [0, 0, dt**2*k_u/(2*m_u), 0, dt]]) * road_sigma/v

    H = np.array([#[-1*c_s/m_s, -1*k_s/m_s, c_s/m_s, k_s/m_s, 0], 
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0,0, 1, 0, 0], 
    [0, 0, 0, 1, 0]])

    error_est = .01 #term for describing inaccuracy of numerical estimate
    if var_v_e is None:
        noise_var_est_sv = np.exp(-4.5 + 58.3 * noise_sigma - 6.2 * v/v_fact - 152.5 * noise_sigma**2 - 65.3*noise_sigma * v/v_fact + 10.3*(v/v_fact)**2)#derived from linear regression during constant velocity simulations
    else:
        noise_var_est_sv = var_v_e
    if var_d_e is None:
        noise_var_est_sd = np.exp(-0.3 + 66.6 * noise_sigma - 20.3 * v/v_fact - 197.7 * noise_sigma**2 - 65.0*noise_sigma * v/v_fact + 18.6*(v/v_fact)**2)
    else:
        noise_var_est_sd = var_d_e
    #print(noise_var_est_sv)
    #print(noise_var_est_sd)
    #sys.exit(0)
    if var_ude is None:
        R = np.array([[noise_sigma, 0, 0], [0, noise_var_est_sv, 0], [0, 0, noise_var_est_sd]])
    else:
        R = np.array([[noise_var_est_sv, 0, 0, 0], [0, noise_var_est_sd, 0, 0], 
        [0, 0, var_uve, 0], [0, 0, 0, var_ude]])
    #print("R is {0}".format(R))
    #R = np.array([])
    #R = np.array([[1, dt/2 + error_est, dt**2/4 + error_est], 
    #[dt/2 + error_est, dt**2/2 + error_est, 3*dt**3/8 + error_est], 
    #[dt**2/4 + error_est, 3*dt**3/8 + error_est, 3*dt**4/8 + error_est]])*noise_sigma 

    return F, H, Q, R

def initialize_kf_sprung_acc(m_s, m_u, c_s, k_u, k_s, road_sigma, noise_sigma, v, dt, F_order=1):
    F = np.array([[-1*c_s/m_s, -1*k_s/m_s, c_s/m_s, k_s/m_s], 
    [1, 0, 0, 0], 
    [c_s/m_u, k_s/m_u, -1*c_s/m_u, -1*(k_u + k_s)/m_u], 
    [0 , 0, 1, 0]])

    F_final = np.eye(len(F))
    for x in range(1, F_order + 1):
        F_final += F*(dt**x)/factorial(x)
    
    Q = np.array([[0, 0, 0, 0], 
    [0, road_sigma, 0, road_sigma], 
    [0, 0, 0, 0],
    [0, road_sigma, 0, road_sigma]])

    H = np.array([[-1*c_s/m_s, -1*k_s/m_s, c_s/m_s, k_s/m_s]])

    R = np.array([[noise_sigma]])
    print("F_final is {0}, H is {1}, Q is {2}, R is {3}".format(F_final, H, Q, R))
    return F_final, H, Q, R






#x0, P0, F, H, Q, R, times, zs, xs, P_preds, P_upds, K_final
def kf_backward(x0, P0, F, H, Q, R, times, zs, x_orig, P_preds, P_upds, K):
    F_t = np.transpose(F)
    xs_smoothed = [x_orig[-1]]
    Ps_smoothed = [P_upds[-1]]
    J_prev = None
    P_t_1_t_2 = np.linalg.multi_dot([np.eye(len(K)) - np.matmul(K, H), F, P_upds[-2]]) # Initialize given last gain
    A = np.zeros(P0.shape)
    B = np.zeros(P0.shape)
    C = np.zeros(P0.shape)
    D = np.zeros(R.shape)
    P_upds.insert(0, P0)
    x_orig.insert(0, x0)
    # print("x0.shape: {0}".format(x0.shape))
    # print("P0.shape: {0}".format(P0.shape))
    # print("F.shape: {0}".format(F.shape))
    # print("H.shape: {0}".format(H.shape))
    # print("Q.shape: {0}".format(Q.shape))
    # print("R.shape: {0}".format(R.shape))
    # print("times.shape: {0}".format(times.shape))
    # print("zs.shape: {0}".format(zs.shape))
    # print("x_orig.shape: {0}".format(len(x_orig)))
    # print("P_preds.shape: {0}".format(len(P_preds)))
    # print("P_upds.shape: {0}".format(len(P_upds)))
    # print("K.shape: {0}".format(K.shape))
    #sys.exit(1)
    for idx in range(len(P_upds) - 2, -1, -1):
        y = zs[idx].reshape(R.shape[0], -1)
        P_t_1_t_1 = P_upds[idx]
        P_t_t_1 = P_preds[idx]
        x_next = xs_smoothed[0]
        P_next = Ps_smoothed[0]
        x_prev_upd = x_orig[idx]
        J = np.linalg.multi_dot([P_t_1_t_1, F_t, np.linalg.inv(P_t_t_1)])
        x_smoothed = x_prev_upd + np.matmul(J, x_next - np.matmul(F, x_prev_upd))
        
        P_smoothed = P_t_1_t_1 + np.linalg.multi_dot([J, P_next - P_t_t_1, np.transpose(J)])
        #xtx = 
        if J_prev is not None:
            #print("I am inSIDE")
            P_t_t = P_upds[idx + 1]
            P_t_1_t_2 = np.matmul(P_t_t, np.transpose(J)) + np.linalg.multi_dot([J_prev, P_t_1_t_2 - np.matmul(F, P_t_t), np.transpose(J)])
        # print("A is {0}".format(A))
        # print("B is {0}".format(B))
        # print("C is {0}".format(C))
        # print("D is {0}".format(D))
        # print('\n\n')
        A +=  P_smoothed + np.matmul(x_smoothed, np.transpose(x_smoothed))
        B += P_t_1_t_2 + np.matmul(x_next, np.transpose(x_smoothed))
        C += P_next + np.matmul(x_next, np.transpose(x_next))
        D += np.matmul(y - np.matmul(H, x_next), np.transpose(y - np.matmul(H, x_next))) + np.linalg.multi_dot([H, P_next, np.transpose(H)])
        
        
        #    P_t_1_t_2 = np.matmul(P_t_1_t_1, np.transpose(J_prev)) + np.linalg.multi_dot([J, P_t_1_t_2 - np.matmul(F, P_t_1_t_1), np.transpose(J_prev)])
        xs_smoothed.insert(0, x_smoothed)
        Ps_smoothed.insert(0, P_smoothed)
        J_prev = np.copy(J)
    # print("A is {0}".format(A))
    # print("B is {0}".format(B))
    # print("C is {0}".format(C))
    # print("D is {0}".format(D))
    #sys.exit(0)
    return xs_smoothed[0], Ps_smoothed[0], xs_smoothed[1:], Ps_smoothed[1:], A, B, C, D
    #TODO: Check formulas tomorrow

def update_constrained(F, Q, A, B, C, M, G, D, n, dt):
    M_t = np.transpose(M)
    A_inv = np.linalg.inv(A)
    try:
        M_t_A_M = np.linalg.inv(np.linalg.multi_dot([M_t, A_inv, M]))
        print("M_t_A_M matrix is {0}".format(np.linalg.multi_dot([M_t, A_inv, M])))

    except Exception as e:
        print("Got singular matrix for M_t_A_M, matrix was {0}".format(np.linalg.multi_dot([M_t, A_inv, M])))
        sys.exit(1)
    #print("F prev was {0}".format(F))
    F_new = np.matmul(B, A_inv)
    #print("F new 1 is {0}".format(F_new))
    #print("C is {0}, n is {1}".format(C, n))
    Q_new = 1/n * (C - np.linalg.multi_dot([B, A_inv, np.transpose(B)]))
    Q_new = (np.transpose(Q_new) + Q_new)#/2 #for numerical stability
    # print("Q was {0}".format(Q))
    print("Q new 1 is {0}".format(Q_new))
    F_M = np.matmul(F_new, M)
    term1 = np.matmul(F_M - G, M_t_A_M)
    F_new = F_new - np.linalg.multi_dot([term1, M_t, A_inv])
    # print("Constraint term added to Q_new is {0}".format(np.matmul(term1, np.transpose(F_M - G))))
    #Q_new = Q_new + np.matmul(term1, np.transpose(F_M - G))
    R_new = 1/n * D
    h_new_r1 = (F_new[0] + np.array([-1, 0, 0, 0]))/dt
    H_new = h_new_r1.reshape(1, -1)
    #H_new = np.vstack([h_new_r1, np.array([1, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0])]) 
    #H_new = H
    # print("F new 2 is {0}".format(F_new))
    # print("Q new 2 is {0}".format(Q_new))
    # #print("H old was {0}".format(H))
    # print("H new is {0}".format(H_new))
    #sys.exit(0)
    return F_new, Q_new, R_new, H_new 


def get_car_list():
    #car1 = qc.QC(m_s=380, m_u=55, k_s=20000, k_u=350000, c_s=2000)
    #car2 = qc.QC(m_s=380, m_u=55, k_s=60000, k_u=350000, c_s=8000)
    car1 = qc.QC(m_s=240, m_u=36, k_s=16000, c_s=980, k_u=160000)
    #car2 = qc.QC(m_s=240, m_u=36, k_s=16000, c_s=980, k_u=160000)
    #car3 = qc.QC(m_s=240, m_u=36, k_s=16000, c_s=2*980, k_u=160000)
    #car4 = qc.QC(m_s=240, m_u=36, k_s=16000, c_s=3*980, k_u=160000)
    car2 = qc.QC(m_s=250, m_u=40, k_s=28000, c_s=2000, k_u=125000)
    car3 = qc.QC(m_s=208, m_u=28, k_s=18709, c_s=1300, k_u=127200)
    car4 = qc.QC(m_s=300, m_u=50, k_s=18000, c_s=1200, k_u=180000)
    car5 = qc.QC(m_s=243, m_u=40, k_s=14671, c_s=370, k_u=124660)
    car6 = qc.QC(m_s=257, m_u=31, k_s=13100, c_s=400, k_u=126000)
    car7 = qc.QC(m_s=290, m_u=59, k_s=16812, c_s=1000, k_u=190000)
    suv1 = qc.QC(m_s=650, m_u=55, k_s=27500, c_s=3000, k_u=237000)
    suv2 = qc.QC(m_s=737.5, m_u=62.5, k_s=26750, c_s=3500, k_u=290000)
    bus = qc.QC(m_s=4500, m_u=500, k_s=300000, c_s=20000, k_u=1600000)
    #nissan = qc.QC(epsilon=7.49803834e+00, omega_s=8.20081439e+00, omega_u=1.99376650e+02, xi=6.50572966e-01)
    #nissan = qc.QC(epsilon=6.03192969e+00, omega_s=1.18815121e+01, omega_u=1.87737414e+02, xi=4.95521488e-01)
    #nissan = qc.QC(epsilon=1.37313204e+01, omega_s=1.25172854e+01, omega_u=1.93174379e+02, xi=3.47041919e-01)
    return [('car1', car1), ('car2', car2), ('car3', car3),  ('car4', car4), 
    ('car5', car5), ('car6', car6), ('car7', car7), ('suv1', suv1), ('suv2', suv2), ('bus', bus)]

def get_true_f(true_car, dt):
    m_s = true_car.m_s
            #divide coefficients by sprung mass to make for easier computation into IRI algorithm
            #Can always get back originals by multiplying by m_s
    c_s = true_car.c * true_car.m_s        #self.c = c_s/self.m_s
    k_u = true_car.k1 * true_car.m_s        #self.k1 = k_u/self.m_s
    k_s = true_car.k2 * true_car.m_s
    m_u = true_car.mu * true_car.m_s       #self.k2 = k_s/self.m_s
            #self.mu = m_u/self.m_s
    true_f =  np.array([[1 - dt*c_s/m_s, -1*k_s/m_s*dt, c_s/m_s*dt, k_s/m_s*dt, 0], 
    [dt, 1, 0, 0, 0], 
    [dt*c_s/m_u, k_s/m_u*dt, 1 - c_s/m_u*dt, -1*(k_u + k_s)/m_u * dt, k_u/m_u * dt], 
    [0 , 0, dt, 1, 0], 
    [0, 0, 0, 0, 1]])
    return true_f


def get_true_q(true_car, v, gn0s, dt):
    road_sigma = gn0s[0]*1e-8/v
    m_s = true_car.m_s
            #divide coefficients by sprung mass to make for easier computation into IRI algorithm
            #Can always get back originals by multiplying by m_s
    c_s = true_car.c * true_car.m_s        #self.c = c_s/self.m_s
    k_u = true_car.k1 * true_car.m_s        #self.k1 = k_u/self.m_s
    k_s = true_car.k2 * true_car.m_s
    m_u = true_car.mu * true_car.m_s       #self.k2 = k_s/self.m_s
            #self.mu = m_u/self.m_s
    Q_true = np.array([[0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0], 
    [0, 0, dt**3*k_u**2/(3*m_u**2), 0, dt**2*k_u/(2*m_u)],
    [0, 0, 0, 0, 0], 
    [0, 0, dt**2*k_u/(2*m_u), 0, dt]]) * road_sigma
    return Q_true
    
# # def compute_sprung_vels(times, accs, init_v, vel, dt, f_c_spatial=200):
#     velocities = np.zeros(len(accs))
#     velocities[0] = init_v
#     #prev_slope = init_slope
#     f_c_time = f_c_spatial * vel
#     alpha = 1/(2*np.pi * f_c_time + 1)
#     for x in range(1, len(velocities)):
#         #dt = times[x] - times[x-1]
#         #C = dt / (3 * (l/vel))  # constant related to 3 * longest wavelength of interest
#         #slopes[x] = C * slopes[x-1] + dt * ((accs[x] - mn))
#         velocities[x] = alpha * velocities[x-1] + (1-alpha)
#     return slopes

# def compute_sprung_disps(times, vels, final_d, vel, f_c_spatial=200):
#     """
#     Returns the estimated displacement of the car as it traveled over the road profile given the slopes
#     :param final_d: The estimated value of the initial displacement
#     :param slopes: The roughness slopes of the road profile
#     :param vels: The list of velocities in m/s
#     :param deltas: The spacing between consecutive measurements
#     :param l: The length of the longest wavelength of interest (in m)
#     :return: The vertical displacements of the car as it traveled over the profile
#     """
#     heights = np.zeros(len(vels))
#     heights[-1] = final_d
#     #vel = np.mean(vels)
#     for x in range(len(heights) - 2, -1, -1):  # loop backwards to try to account for phase shift
#         dt = times[x+1] - times[x]
#         C = dt / (3 * (l/vel))  # drift removal term
#         heights[x] = C * heights[x + 1] + dt * vels[x]
#     return heights

def euler_approx(times, true_elevations, true_car, true_vels, true_disps, us_vels, us_disps, st_index=99):
    dt = times[1] - times[0]
    m_s = true_car.m_s
            #divide coefficients by sprung mass to make for easier computation into IRI algorithm
            #Can always get back originals by multiplying by m_s
    c_s = true_car.c * true_car.m_s        #self.c = c_s/self.m_s
    k_u = true_car.k1 * true_car.m_s        #self.k1 = k_u/self.m_s
    k_s = true_car.k2 * true_car.m_s
    m_u = true_car.mu * true_car.m_s  

    F = np.array([[-1*c_s/m_s, -1*k_s/m_s, c_s/m_s, k_s/m_s, 0], 
    [1, 0, 0, 0, 0], 
    [c_s/m_u, k_s/m_u, -1*c_s/m_u, -1*(k_u + k_s)/m_u , k_u/m_u], 
    [0 , 0, 1, 0, 0], 
    [0, 0, 0, 0, 0]]) * dt
    # F = np.array([[1 - dt*c_s/m_s, -1*k_s/m_s*dt, c_s/m_s*dt, k_s/m_s*dt, 0], 
    # [dt, 1, 0, 0, 0], 
    # [dt*c_s/m_u, k_s/m_u*dt, 1 - c_s/m_u*dt, -1*(k_u + k_s)/m_u * dt, k_u/m_u * dt], 
    # [0 , 0, dt, 1, 0], 
    # [0, 0, 0, 0, 1]])
    #F = expm(F)
    F = np.eye(len(F)) + F #+ np.matmul(F, F)/2 #+ np.linalg.multi_dot([F, F, F])/6
    print("Second term is {0}".format(np.matmul(F, F)/2))
    print("Third term is {0}".format(np.linalg.multi_dot([F, F, F])/6))
    print("Fourth term is {0}".format(np.linalg.multi_dot([F, F, F, F])/24))
    
    x0 = np.array([[true_vels[st_index]], [true_disps[st_index]], [us_vels[st_index]], [us_disps[st_index]], [true_elevations[st_index]/1000]])
    #x0 = np.array([[true_vels[st_index]], [true_disps[st_index] - true_elevations[st_index]], [us_vels[st_index]], [us_disps[st_index] - true_elevations], [true_elevations[st_index]/1000]])

    x_prev = x0
    vels, disps, us_vels, us_disps = [], [], [], []
    for el in true_elevations[st_index+1:]:
        x_next = np.matmul(F, x_prev)
        #print("X_next is {0}\n".format(x_next))
        x_next[4][0] = el/1000
        vels.append(x_next[0][0])
        disps.append(x_next[1][0])
        us_vels.append(x_next[2][0])
        us_disps.append(x_next[3][0])
        x_prev = x_next
    return vels, disps, us_vels, us_disps
    

def maximize_likelihood(true_car, init_car_params, profile_len, seg_lens, prof_types, dx, start_seed, velocity, sample_rate_hz,
noise_sigma=1e-2, max_iters = 50, eps=1e-1):
    profile, gn0s = generate_road_profile(profile_len, seg_lens, prof_types, dx, start_seed)

    times, accs, true_elevations, true_vels, true_disps, us_vels, us_disps = drive_over_profile(true_car, profile, velocity, sample_rate_hz)
    
    euler_vels, euler_disps, euler_us_vels, euler_us_disps = euler_approx(times, true_elevations, true_car, true_vels, true_disps, us_vels, us_disps)
    
    #plt.plot(detrend(cumtrapz(accs, times, initial=0)), label='Int Accs')
    # plt.plot(detrend(cumtrapz(true_vels, times, initial=0)), label='Int Vels')
    #pl_ed = len()
    plt.plot(true_vels[100:], label='True_Vels')
    plt.plot(euler_vels, label='Euler vels, MSE = {0}'.format(np.sum((euler_vels - true_vels[100:])**2)/len(euler_vels)))
    plt.legend()
    plt.show()
    
    plt.plot(true_disps[100:], label='True_Disps')
    plt.plot(euler_disps, label='Euler Disps, MSE = {0}'.format(np.sum((euler_disps - true_disps[100:])**2)/len(euler_vels)))
    plt.legend()
    plt.show()
    plt.plot(us_vels[100:], label='True_Us_Vels')
    plt.plot(euler_us_vels, label='Euler_Us_vels, MSE = {0}'.format(np.sum((euler_us_vels - us_vels[100:])**2)/len(euler_vels)))
    plt.legend()
    plt.show()
    plt.plot(us_disps[100:], label='True_Us_Disps')
    plt.plot(euler_us_disps, label='Euler_Us_Disps, MSE = {0}'.format(np.sum((euler_us_disps - us_disps[100:])**2)/len(euler_vels)))
    plt.legend()
    plt.show()
    # plt.legend()
    # plt.show()
    # #sys.exit(0)
    # #plt.plot(true_vels, label='True_Vels')
    # plt.plot(true_disps, label='True_Disps')
    # plt.legend()
    # plt.show()
    # sys.exit(0)

    
    #f, pxx = periodogram(true_elevations/1000, fs=sample_rate_hz)
    #f_road, pxx_road = welch(profile.get_elevations()/1000, fs=1/dx, nperseg=2048, noverlap=1024)
    #f, pxx = welch(true_elevations/1000, fs=sample_rate_hz, nperseg=1024, noverlap=512)
    #f_time_est = f_road * velocity
    #pxx_est = gn0s[0] * 1e-8 * (f_time_est)**-2 * velocity
    #plt.loglog(f, pxx, label='Road PSD, Time Domain')
    #plt.loglog(f_road, pxx_road, label='Road PSD, Spatial')
    #plt.loglog(f_time_est, pxx_est, label='Hyp Road PSD, Time')
    #plt.loglog()
    #plt.loglog(f_road*velocity, pxx_road/velocity, label='Theroetical Periodogram')
    #plt.loglog(f_est*velocity, pxx_est, label='Theroetical Periodogram2')
    #plt.loglog(f_orig*velocity, pxx_est * velocity, label='Theroetical Periodogram3')
    #plt.legend()
    #plt.title("Road profile PSD Time Domain")
    #plt.show()
    # print("Gno is {0}".format(gn0s))
    # #print("Var is {0}".format(np.var(profile.get_elevations()/1000)/(gn0s[0]*1e-3*.01)))
    # print("Var profile is {0}".format(np.var(profile.get_elevations()/1000)))
    # pred_var = gn0s[0] * 1e-6 * (1/(.1))**-2 * dx
    # print("Predicted var profile is {0}".format(pred_var))
    # rp_var = np.var(np.diff(profile.get_elevations()/1000))
    # f, pxx = welch(profile.get_elevations()/1000, fs=1/dx, nperseg=2048, noverlap=1024)
    # f_per, pxx_per = periodogram(profile.get_elevations()/1000, fs=1/dx)
    # f2 = np.arange(1e-5, (1/dx)/2, .1)
    # pxx2 = (f2)**(-2)* (gn0s[0] * 1e-6 * (1/(.1))**-2)
    # plt.loglog(f, pxx, label='Welch PSD')
    # plt.loglog(f2, pxx2, label='Theoretical PSD')
    # plt.loglog(f_per, pxx_per, label='Periodogram')
    # plt.xlim(.01, 10)
    # plt.legend()
    # plt.show()
    #f_diffw, pxx_diffw = welch(np.diff(true_elevations/1000), fs=1/sample_rate_hz, nperseg=1024, noverlap=512)
    #psd_est = len(f_diffw) * [gn0s[0] * 1e-8 * velocity]
    #plt.loglog(f_diffw, pxx_diffw, label='Welch Diff PSD')
    #plt.loglog(f_diffw, psd_est, label='Hyp PSD')
    #plt.loglog(f_diffw, [np.var(np.diff(true_elevations/1000))]*len(f_diffw), label='Var PSD')
    #print(np.median(pxx_diffw)/psd_est[0])
    #plt.legend()
    #plt.show()
    
    # print("Var diff is {0}".format(rp_var))
    
    # print("Predicted var diff is {0}".format(pred_var))
    # print("Var ratio is {0}".format(rp_var/pred_var))
    # print("Welch says {0}".format(np.median(pxx_diffw[np.where(f_diffw <= 10)])))
    # f_diff_per, pxx_diff_per = periodogram(np.diff(profile.get_elevations()/1000), fs=1/dx)
    # plt.plot(f_diffw, pxx_diffw, label='Welch')
    # #plt.plot(f_diff_per, pxx_diff_per, label='Periodogram')
    # plt.plot(f2, np.zeros(len(f2)) + rp_var, label='Computed Variance')
    # plt.plot(f2, np.zeros(len(f2)) + pred_var*dx, label='Pred Variance')
    # plt.xlim(0, 15)
    # plt.legend()
    # plt.show()
    # sys.exit(0)
    #f, pxx = periodogram(np.diff(profile.get_elevations()), fs=1/dx)
    #plt.hlines(gn0s[0] * 1e-6 * 1000 , xmin=0, xmax=15)
    #plt.plot(f, pxx)
    #plt.show()
    #plt.hist(np.diff(profile.get_elevations()))
    #plt.show()
    print(len(times), len(true_elevations))
    #sys.exit(0)
    #Add Noise
    #accs += np.random.normal(loc=0, scale=np.sqrt(noise_sigma), size=accs.shape)
    times = times[100:]
    accs = accs[100:]
    #x0 = np.array([[true_vels[99]], [true_disps[99]], [us_vels[99]], [us_disps[99]], [true_elevations[99]/1000]]) #sprung velocity
    #x0 = np.array([[0], [0], [0], [0], [0]]) #Initial estimate - all zeros
    #P0 = np.eye(5)*.0001 # just set this to identity
    x0 = np.array([[true_vels[99]], [true_disps[99] - true_elevations[99]/1000], [us_vels[99]], [us_disps[99] - true_elevations[99]/1000]])
    P0 = np.eye(4) * .0001
    
    np.random.seed(start_seed)
    noise = np.random.normal(loc=0, scale=np.sqrt(noise_sigma), size=accs.shape)
    accs = accs + noise

    wave_c = 100
    time_cutoff_freq = 1/wave_c * velocity # 
    filter_order = 6
    sos = butter(filter_order, time_cutoff_freq, btype='highpass', analog=False, output='sos', fs=sample_rate_hz)
    noise2 = np.random.normal(loc=0, scale=np.sqrt(3*noise_sigma), size=accs.shape)
    noise3 = np.random.normal(loc=0, scale=np.sqrt(5*noise_sigma), size=accs.shape)
    noise4 = np.random.normal(loc=0, scale=np.sqrt(2*noise_sigma), size=accs.shape)
    noise5 = np.random.normal(loc=0, scale=np.sqrt(noise_sigma), size=accs.shape)
    vels = true_vels[100:] + noise2
    disps = true_disps[100:] + noise3
    us_vels_noise = us_vels[100:] + noise4
    us_disps_noise = us_disps[100:] + noise5
    #vels = sosfilt(sos, cumtrapz(accs, times, initial=0))
    #vs_non_noise = cumtrapz(accs,times, )
    #disps = sosfilt(sos, cumtrapz(vels, times, initial=0))
    #compute_sprung_vels(times, accs + noise, 0, velocity, l=100, centered=False)
    #disps = cumtrapz(vels, times, initial=0)#compute_sprung_disps(times, vels, 0, velocity, l=.04)
    #vels4 = compute_sprung_vels(times, accs + noise, 0, velocity, l=100)
    #vels2 = detrend(cumtrapz(accs, times, initial=0))
    #disps2 = detrend(cumtrapz(vels2, times, initial=0))
    #plt.plot(accs, label='accs')
    #plt.plot(vels, label='est_vels')
    #plt.plot(disps, label='est_disps')
    #plt.plot(true_vels[100:], label='true_vels')
    #plt.plot(true_disps[100:], label='true_disps')
    var_v_e = np.var(true_vels[100:] - vels)
    var_d_e = np.var(true_disps[100:] - disps)
    print("True vel var error is {0}".format(var_v_e))
    print("True disp var error is {0}".format(var_d_e))

    #plt.plot(vels2, label='vels2')
    #plt.plot(disps2, label='disps2')
    #plt.plot(vels4, label='vels4')
    # plt.plot(true_elevations/1000, label='road_profile')
    # plt.legend()
    # plt.show()
    # probplot(accs - noise, dist="norm", plot=plt)
    # plt.title("Prob plot acc errors")
    # plt.show()
    # plt.hist(true_vels[100:] - vels)
    # plt.title("hist of velocity errors")
    # plt.show()
    # probplot(true_vels[100:] - vels, dist="norm", plot=plt)
    # plt.title("Prob plot vel errors")
    # plt.show()
    # probplot(true_disps[100:] - disps, dist="norm", plot=plt)
    # plt.title("Prob plot disp errors")
    # plt.show()
    # plt.hist(true_disps[100:] - disps)
    # plt.title("hist of disp errors")
    # plt.show()
    #Measurement = sprung velocity, sprung displacement, unsprung velocity, unsprung displacement
    #zs = np.transpose(np.vstack([vels, disps, us_vels_noise, us_disps_noise]))

    #Measurement = sprung acceleration
    zs = accs.reshape(-1, 1)

    # plt.plot(accs, label='IDK')
    # plt.plot(zs[:, 0], label='Please')
    # plt.legend()
    # plt.show()
    # plt.plot(vels, label='IDK')
    # plt.plot(zs[:, 1], label='Please')
    # plt.show()
    # plt.plot(disps, label='IDK')
    # plt.plot(zs[:, 2], label='Please')
    # plt.show()
    #TODO: Also, look at using the IRI algorithm for integrating the accelerations
    curr_l = None
    dt=1/sample_rate_hz
    
    #4 measurements
    #F, H, Q, R = initialize_kf(**init_car_params, road_sigma=gn0s[0]*1e-8, noise_sigma=noise_sigma, v=velocity, dt=dt, 
    #var_v_e=var_v_e, var_d_e=var_d_e, var_uve=np.var(us_vels_noise - noise4), var_ude=np.var(us_disps_noise - noise5))
    
    #1 measurement
    F, H, Q_orig, R = initialize_kf_sprung_acc(**init_car_params, road_sigma=np.var(np.diff(true_elevations/1000)), noise_sigma=noise_sigma,
    v=velocity, dt=dt)
    Q = Q_orig
    #Q = np.eye(len(Q))
    curr_it = 0
    #M = np.array([[1], [1], [1], [1], [1]])
    #G = np.array([[1], [dt + 1], [1], [dt + 1], [1]])
    #M = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 1]])
    #G = np.array([[1, 0, 1], [dt, 1, dt + 1], [1, 0, 1], [dt, 1, dt + 1], [0, 1, 1]])
    k_u = true_car.k1 * true_car.m_s
    m_u = true_car.mu * true_car.m_s
    M = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    G = np.array([[1, 0], [dt, 1], [1, -1*k_u/m_u*dt], [dt, 1]])
    log_ls = []
    log_ls2 = []
    print("Road profile variance: {0}".format(np.var(np.diff(true_elevations/1000))))
    
    #Q = np.zeros((5, 5))
    #Q = np.eye(5)*.01
    #Q[-1][-1] = np.var(np.diff(true_elevations/1000))/2
    print("Q is {0}".format(Q))
    #sys.exit(0)
    #print(np.sum(gn0s[0] * 1e-8 * velocity))
    kf_filt = KalmanFilter(dim_x=4, dim_z=1)
    kf_filt.x = x0
    kf_filt.P = P0
    kf_filt.F = F
    kf_filt.R = R
    kf_filt.H = H
    #Q_cwn = Q_continuous_white_noise(dim=1, dt=1/sample_rate_hz, spectral_density=gn0s[0]) 
    #print("Continuous white noise Q: {0}, our Q: {1}".format(Q_cwn, Q))
    kf_filt.Q = Q
    while True:
        #x0, P0, F, H, Q, R, times, zs
        xs, P_preds, P_upds, log_l, K_final = kf_forward(x0, P0, F, H, Q, R, times, zs)
        #xs_kf, Ps_kf, xps_kf, Pps_kf = kf_filt.batch_filter(zs)
        xs_kf, Ps_kf, xps_kf, Pps_kf = [], [], [], []
        kf_lls = []
        for i, z in enumerate(zs):

            kf_filt.predict(u=None, B=None, F=F, Q=Q)
            xps_kf.append(kf_filt.x)
            Pps_kf.append(kf_filt.P)

            kf_filt.update(z, R=R, H=H)
            xs_kf.append(kf_filt.x)
            Ps_kf.append(kf_filt.P)
            kf_lls.append(kf_filt.log_likelihood)
        kf_filt.x = x0
        kf_filt.P = P0
        kf_filt.F = F
        kf_filt.R = R
        kf_filt.H = H
        kf_filt.Q = Q
        xs_kf, Ps_kf, xps_kf, Pps_kf = kf_filt.batch_filter(zs)
        log_l2 = np.sum(kf_lls)
        #print("My LL: {0}, filterpy LL: {1}".format(log_l, np.sum(kf_lls) * 2*np.pi))
        #sys.exit(0)
        # plt.plot(accs, label="Measured Accs")
        # accs_est = []
        # for x in xs:
        #     accs_est.append(np.matmul(H, x)[0][0])
        # plt.plot(accs_est, label='Pred Accs')
        # plt.legend()
        # plt.show()
        plt.plot(list(map(lambda x: x[0][0], xs)), label='Pred Svs')
        plt.plot(true_vels[100:], label='True Svs')
        plt.legend()
        plt.show()
        
        #plt.plot(list(map(lambda x: x[1][0], xs)), label='Pred SDs')
        plt.plot(np.array(list(map(lambda x: x[1][0], xs))), label='Pred SDs - ys')
        #plt.plot(true_disps[100:], label='True SDs')
        plt.plot(true_disps[100:] - true_elevations[100:]/1000, label='True SDs - ys')
        plt.legend()
        plt.show()

        plt.plot(list(map(lambda x: x[2][0], xs)), label='Pred Usvs')
        plt.plot(us_vels[100:], label='True USvs')
        plt.legend()
        plt.show()

        #plt.plot(list(map(lambda x: x[3][0], xs)), label='Pred USvs')
        #plt.plot(us_disps, label='True USds')
        plt.plot(np.array(list(map(lambda x: x[3][0], xs))), label='Pred USvs - ys')
        plt.plot(us_disps[100:] - true_elevations[100:]/1000, label='True USds - ys')
        plt.legend()
        plt.show()
        #sys.exit(0)
        # plt.plot(times, true_elevations[100:], label='True Prof')
        # print(np.allclose(np.array(xs), np.array(xs_kf)))
        # print(np.allclose(np.array(P_upds), np.array(Ps_kf)))
        # plt.plot(times, list(map(lambda x: x[4][0]*1000, xs)), label='Est_Prof')
        # plt.plot(times, list(map(lambda x: x[4][0]*1000, xs_kf)), label='Est_Prof_FPY')
        # plt.legend()
        # plt.ylim(-50, 50)
        # plt.title("True vs Est Profile, forward")
        # plt.show()
        if len(log_ls) > 0:
            print("Prev ll: {0}, curr ll: {1}".format(log_ls[-1], log_l))
            print("Prev ll2: {0}, curr ll2: {1}\n".format(log_ls2[-1], log_l2))
            
        log_ls.append(log_l)
        log_ls2.append(log_l2)
        if (curr_l is not None and abs(log_l - curr_l) <= eps) or curr_it >= max_iters:
            break
        
        #xs_smoothed[0], Ps_smoothed[0], xs_smoothed[1:], Ps_smoothed[1:], A, B, C, D
        x0, P0, x_smoothed, P_smoothed, A, B, C, D = kf_backward(x0, P0, F, H, Q, R, times, zs, xs, P_preds, P_upds, K_final)
        xs_kf_smoothed, Ps_kf_smoothed, _, _ = kf_filt.rts_smoother(xs_kf, Ps_kf)
        #print(len(xs_kf_smoothed))
        #print(len(x_smoothed))
        #print("R prior update: {0}".format(R))
        #print("Q prior update: {0}\n".format(Q))
        #F, Q, A, B, C, M, G, D, n
        #print("len x smoothed: {0}".format(len(x_smoothed)))
        print("H old was {0}".format(H))
        F, _, _, H = update_constrained(F, Q, A, B, C, M, G, D, len(x_smoothed), 1/sample_rate_hz)
        #print("R after update: {0}".format(R))
        #print("Q after update: {0}\n".format(Q))
        curr_l = log_l
        curr_it += 1
        # plt.plot(times, true_elevations[100:], label='True Prof')
        # print(np.allclose(np.array(x_smoothed), np.array(xs_kf_smoothed)))
        # print(np.allclose(np.array(P_smoothed), np.array(Ps_kf_smoothed)))
        # plt.plot(times, list(map(lambda x: x[4][0]*1000, x_smoothed)), label='Est_Prof')
        # plt.plot(times, list(map(lambda x: x[4][0]*1000, xs_kf_smoothed)), label='Est_Prof_FPY')
        # plt.title("True vs. Est Profile, Smoothed it = {0}".format(curr_it))
        # plt.ylim(-50, 50)
        # plt.legend()
        # plt.show()
        kf_filt = KalmanFilter(dim_x=4, dim_z=1)
        kf_filt.x = x0
        kf_filt.P = P0
        kf_filt.F = F
        kf_filt.R = R
        kf_filt.H = H
        #Q_cwn = Q_continuous_white_noise(dim=1, dt=1/sample_rate_hz, spectral_density=gn0s[0]) 
        #print("Continuous white noise Q: {0}, our Q: {1}".format(Q_cwn, Q))
        #kf_filt.Q = Q
        kf_filt.Q = Q
    plt.plot(range(1, len(log_ls) + 1), log_ls)
    plt.title("Log Likelihood for {0} iterations".format(len(log_ls)))
    plt.show()
    plt.plot(range(1, len(log_ls2) + 1), log_ls2)
    plt.title("Log Likelihood Fpy for {0} iterations".format(len(log_ls)))
    plt.show()
    print("F ML: {0}".format(F))
    print("True F: {0}".format(get_true_f(true_car, dt)))
    print('\n')

    print("Q ML: {0}".format(Q))
    print("True Q: {0}".format(Q_orig))
    print("\n\n\n")
    true_iri = profile.to_iri()
    est_els = np.array(list(map(lambda x: x[4][0]*1000, x_smoothed)))
    est_prof = rp.RoadProfile(times*velocity, est_els)
    est_iri = est_prof.to_iri()
    print("True IRI: {0}, est IRI: {1}".format(true_iri, est_iri))
    print("Est Prof MSE: {0}".format(np.sum(true_elevations[100:] - est_els)**2/len(est_els)))

def run_exp(num_cars, prof_len=500):
    init_car_params = {"m_s": 400, "c_s": 900, "k_s": 15000, "k_u": 130000, "m_u": 35}
    for c in get_car_list()[:num_cars]:
        #init_car_params = {"m_s": c[1].m_s, "c_s": c[1].c*c[1].m_s, 
        #"k_s": c[1].k2*c[1].m_s, "k_u": c[1].k1*c[1].m_s, "m_u": c[1].mu*c[1].m_s}
        #print("init car params: {0}".format(init_car_params))
        print("Maximizing likelihood for car: {0}, length = {1}".format(c[0], prof_len))
        #sys.exit(0)
        maximize_likelihood(c[1], init_car_params, prof_len, [prof_len], ['A'], .05, 1, 10, 100, eps=1, noise_sigma=.001)

run_exp(1, 1000)
#for c in get_car_list():
#    print("ku/mu is {0}".format(np.sqrt(c[1].k1/c[1].mu)))
