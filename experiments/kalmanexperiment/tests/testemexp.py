from experiments.kalmanexperiment import emexp
from quartercar.roadprofile import RoadProfile
from quartercar import cars, qc
import numpy as np
from scipy.linalg import expm



def setup(sample_rate=100):
    fname = '/Users/gregoryislas/Documents/Mobilized/data_dump/Road_Profiles_Test/Profile_A_500_0'

    dists = np.load("{0}_distances.npy".format(fname))
    elevations =  np.load("{0}_elevations.npy".format(fname))

    prof = RoadProfile(distances=dists, elevations=elevations)
    test_car = cars.get_car_list()[1][1]

    test_vel = 10
    orig_sr_hz = 1000
    T, yout, xout, new_dists, new_els, vs = test_car.run2(prof, [], test_vel, final_sample_rate=orig_sr_hz)
    #road_sigma = np.var(np.diff(new_els/1000))
    sprung_accs = yout[:, -1]
    #sample_rate = 200
    step = int(orig_sr_hz/sample_rate)

    sprung_accs = sprung_accs[list(range(0, len(sprung_accs), step))]

    sprung_vels = xout[:, 1] #sprung velocities
    sprung_vels = sprung_vels[list(range(0, len(sprung_vels), step))]

    sprung_disps = xout[:, 0] #you get the idea
    sprung_disps = sprung_disps[list(range(0, len(sprung_disps), step))]

    unsprung_vels = xout[:, 3]
    unsprung_vels = unsprung_vels[list(range(0, len(unsprung_vels), step))]

    unsprung_disps = xout[:, 2]
    unsprung_disps = unsprung_disps[list(range(0, len(unsprung_disps), step))]

    prof_input = new_els[list(range(0, len(new_els), step))]/1000
    road_sigma = np.var(np.diff(prof_input))
    acc_noise_level = .01
    noise = np.random.normal(loc=0, scale=np.sqrt(acc_noise_level), size=sprung_accs.shape)

    accs_plus_noise = sprung_accs + noise
    x0 = np.array([[0], [0], [0], [0]])
    P0 = np.eye(4) * .01
    return test_car, x0, P0, accs_plus_noise, acc_noise_level, road_sigma, sample_rate, test_vel, prof_input

def test_basic_initialize_kf_est_params():
    test_car, x0, P0, accs_plus_noise, acc_sigma, road_sigma, sample_rate,test_vel, prof_input = setup()
    dt = 1/sample_rate
    Fknown = 1
    Qknown = 1
    F_test, H_test, Q_test, R_test, M_test, G_test, p_to_opt_test = emexp.initialize_kf_est_params(test_car, road_sigma, Fknown, Qknown, None, acc_sigma, dt, order=1)
    F_shouldbe = np.array([[-1*test_car.c, -1*test_car.k2, test_car.c, test_car.k2], 
[1, 0, 0, 0], [test_car.c/test_car.mu, test_car.k2/test_car.mu, -1*test_car.c / test_car.mu, -(test_car.k1 + test_car.k2) / test_car.mu],
[0, 0, 1, 0]]) * dt + np.eye(4)
    H_shouldbe = np.array([[-1*test_car.c, -1*test_car.k2, test_car.c, test_car.k2]])
    Q_shouldbe = np.array([[0, 0, 0, 0], [0, road_sigma, 0, road_sigma], [0, 0, 0, 0], [0, road_sigma, 0, road_sigma]])
    R_shouldbe = np.array([[acc_sigma]])
    
    assert(np.allclose(F_test, F_shouldbe))
    assert(np.allclose(H_test, H_shouldbe))
    assert(np.allclose(Q_test, Q_shouldbe))
    assert(np.allclose(R_test, R_shouldbe))
    assert(p_to_opt_test[0] == 'x0')
    assert(p_to_opt_test[1] == 'P0')

def test_basic_estimate_model_parameters():
    test_car, x0, P0, accs_plus_noise, acc_sigma, road_sigma, sample_rate, test_vel, prof_input  = setup()
    Fknown = 1
    Qknown = 1
    dt = 1/sample_rate
    ret_dict_avg, ret_dict_max, ret_dict_var = emexp.estimate_model_parameters(test_car, road_sigma, acc_sigma, test_vel, accs_plus_noise.reshape(-1, 1), dt, Fknown, Qknown, 
    1, 2, 1, x0, P0)
    true_eps = 1/test_car.mu
    true_ws = np.sqrt(test_car.k2)
    true_wu = np.sqrt(test_car.k1/test_car.mu)
    true_xi = test_car.c/(2 * true_ws)
    assert(np.isclose(true_eps, ret_dict_avg['Est_Eps_Avg']))
    assert(np.isclose(true_eps, ret_dict_max['Est_Eps_Max']))

    assert(np.isclose(true_ws, ret_dict_avg['Est_Ws_Avg']))
    assert(np.isclose(true_ws, ret_dict_max['Est_Ws_Max']))

    assert(np.isclose(true_wu, ret_dict_avg['Est_Wu_Avg']))
    assert(np.isclose(true_wu, ret_dict_max['Est_Wu_Max']))

    assert(np.isclose(true_xi, ret_dict_avg['Est_Xi_Avg']))
    assert(np.isclose(true_xi, ret_dict_max['Est_Xi_Max']))


    assert(np.isclose(0, ret_dict_var['Var_Eps']))
    assert(np.isclose(0, ret_dict_var['Var_Wu']))
    assert(np.isclose(0, ret_dict_var['Var_Ws']))
    assert(np.isclose(0, ret_dict_var['Var_Xi']))
    assert(np.isclose(0, ret_dict_var['Var_Road_Sigma']))

def test_ret_shape_estimate_model_parameters():
    test_car, x0, P0, accs_plus_noise, acc_sigma, road_sigma, sample_rate, test_vel, prof_input  = setup()
    Fknown = 1
    Qknown = 1
    dt = 1/sample_rate
    ret_dict_avg, ret_dict_max, ret_dict_var = emexp.estimate_model_parameters(test_car, road_sigma, acc_sigma, test_vel, accs_plus_noise.reshape(-1, 1), dt, Fknown, Qknown, 
    1, 2, 1, x0, P0)
    est_xs_avg = ret_dict_avg['Avg_Xs']
    est_xs_max = ret_dict_max['Max_Xs']
    #assert(est_xs_avg[0].shape == est_xs_max.shape)
    assert(np.allclose(est_xs_avg, np.array(est_xs_max)))
    zs_prof = np.hstack([accs_plus_noise.reshape(-1, 1), np.transpose(np.hstack(est_xs_avg))]) 
    assert(zs_prof.shape == (len(accs_plus_noise), 5))

def get_mean(list_arrays, x, y):
    num, denom = 0, 0
    for l in list_arrays:
        num += l[x][y]
        denom += 1
    return num/denom

def test_initialize_prof_est_kf():
    test_car, x0, P0, accs_plus_noise, acc_sigma, road_sigma, sample_rate, test_vel, prof_input  = setup()
    Fknown = 1
    Qknown = 1
    dt = 1/sample_rate
    ret_dict_avg, ret_dict_max, ret_dict_var = emexp.estimate_model_parameters(test_car, road_sigma, acc_sigma, test_vel, accs_plus_noise.reshape(-1, 1), dt, Fknown, Qknown, 
    1, 2, 1, x0, P0)
    est_eps = ret_dict_avg['Est_Eps_Avg']
    est_ws = ret_dict_avg['Est_Ws_Avg']
    est_wu = ret_dict_avg['Est_Wu_Avg']
    est_xi = ret_dict_avg['Est_Xi_Avg']
    #est_x0 = ret_dict_avg['Avg_X0']
    #est_P0 = ret_dict_avg['Avg_P0']
    est_Ps = ret_dict_avg['Avg_Ps']
    F_shouldbe = np.array([[-1*test_car.c, -1*test_car.k2, test_car.c, test_car.k2, 0], 
[1, 0, 0, 0, 0], [test_car.c/test_car.mu, test_car.k2/test_car.mu, -1*test_car.c / test_car.mu, -(test_car.k1 + test_car.k2) / test_car.mu, test_car.k1/test_car.mu],
[0, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
    H_shouldbe = np.array([[-1*test_car.c, -1*test_car.k2, test_car.c, test_car.k2, 0], [1,  0, 0, 0, 0], 
    [0, 1, 0, 0, -1], [0, 0, 1, 0, 0], [0, 0, 0, 1, -1]])
    F_final, H_final, Q_final, R_prof = emexp.initialize_kf_est_profile(est_eps, est_ws, est_wu, est_xi, road_sigma, est_Ps, acc_sigma, 1/sample_rate)
    Q_shouldbe = np.zeros((5, 5))
    Q_shouldbe[-1][-1] = road_sigma
    assert(np.allclose(expm(F_shouldbe*dt), F_final))
    assert(np.allclose(H_shouldbe, H_final))
    assert(np.allclose(Q_final, Q_shouldbe))
    Rshouldbe = np.zeros((5, 5))
    Rshouldbe[0][0] = acc_sigma
    for x in range(1, 5):
        for y in range(1, 5):
            Rshouldbe[x][y] = get_mean(est_Ps, x-1, y-1)
    assert(np.allclose(Rshouldbe, R_prof))

def test_estimate_profile():
    test_car, x0, P0, accs_plus_noise, acc_sigma, road_sigma, sample_rate, test_vel, prof_input  = setup()
    Fknown = 1
    Qknown = 1
    dt = 1/sample_rate
    ret_dict_avg, ret_dict_max, ret_dict_var = emexp.estimate_model_parameters(test_car, road_sigma, acc_sigma, test_vel, accs_plus_noise.reshape(-1, 1), dt, Fknown, Qknown, 
    1, 2, 1, x0, P0)
    est_eps = ret_dict_avg['Est_Eps_Avg']
    est_ws = ret_dict_avg['Est_Ws_Avg']
    est_wu = ret_dict_avg['Est_Wu_Avg']
    est_xi = ret_dict_avg['Est_Xi_Avg']
    est_x0 = ret_dict_avg['Avg_X0']
    est_P0 = ret_dict_avg['Avg_P0']
    est_Ps = ret_dict_avg['Avg_Ps']
    est_xs = ret_dict_avg['Avg_Xs']
    est_prof = emexp.estimate_profile(est_eps, est_ws, est_wu, est_xi, road_sigma, est_x0, est_P0,  est_xs, est_Ps, accs_plus_noise.reshape(-1, 1), acc_sigma, dt)
    assert(len(prof_input) == len(est_prof))


    


#test_basic_estimate_model_parameters()
    


