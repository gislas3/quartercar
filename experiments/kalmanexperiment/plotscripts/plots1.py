from experiments.kalmanexperiment import emexp
from quartercar.roadprofile import RoadProfile
from quartercar import cars, qc
import numpy as np
from scipy.linalg import expm
from matplotlib import pyplot as plt

fname = '/Users/gregoryislas/Documents/Mobilized/data_dump/Road_Profiles_Test/Profile_A_500_0'

dists = np.load("{0}_distances.npy".format(fname))
elevations =  np.load("{0}_elevations.npy".format(fname))

prof = RoadProfile(distances=dists, elevations=elevations)
test_car = cars.get_car_list()[4][1]

test_vel = 10
orig_sr_hz = 1000
T, yout, xout, new_dists, new_els, vs = test_car.run2(prof, [], test_vel, final_sample_rate=orig_sr_hz)
#road_sigma = np.var(np.diff(new_els/1000))
sprung_accs = yout[:, -1]
sample_rate = 200
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
ret_dict_avg, ret_dict_max, ret_dict_var = emexp.estimate_model_parameters(test_car, road_sigma, acc_noise_level, test_vel, accs_plus_noise.reshape(-1, 1), 1/sample_rate, 1, 1, 
1, 2, 1, x0, P0)

true_eps = 1/test_car.mu
true_ws = np.sqrt(test_car.k2)
true_wu = np.sqrt(test_car.k1/test_car.mu)
true_xi = test_car.c/(2 * true_ws)
print("True epsilon is {0}, est epsilon is {1}".format(true_eps, ret_dict_avg['Est_Eps_Avg']))

print("True ws is {0}, est ws is {1}".format(true_ws, ret_dict_avg['Est_Ws_Avg']))
print("True wu is {0}, est wu is {1}".format(true_wu, ret_dict_avg['Est_Wu_Avg']))
print("True xi is {0}, est xi is {1}".format(true_xi, ret_dict_avg['Est_Xi_Avg']))


est_xs = ret_dict_avg['Avg_Xs']
print(est_xs.shape)
print(len(ret_dict_max['Max_Xs']), ret_dict_max['Max_Xs'][0].shape)
sprung_vels_est = est_xs[:, 0, 0]
sprung_disps_est = est_xs[:, 1, 0]

unsprung_vels_est = est_xs[:, 2, 0]

unsprung_disps_est = est_xs[:, 3, 0]

fact = 1
plt.plot(sprung_vels, label='Sprung Vels True')
plt.plot(sprung_vels_est*fact, label='Sprung Vels Est')
plt.legend()
plt.show()

plt.plot(sprung_vels - sprung_vels_est*fact, label='Sprung Vels True - Sprung Vels Est')
plt.legend()
plt.show()

plt.plot(sprung_disps - prof_input, label='Sprung Disps True')
plt.plot(sprung_disps_est, label='Sprung Disps Est')
plt.legend()
plt.show()

plt.plot(unsprung_vels, label='Unsprung Vels True')
plt.plot(unsprung_vels_est, label='Unsprung Vels Est')
plt.legend()
plt.show()

plt.plot(unsprung_disps - prof_input, label='Unsprung Disps True')
plt.plot(unsprung_disps_est, label='Unsprung Disps Est')
plt.legend()
plt.show()

# plt.plot(sprung_vels, label='Sprung Vels True')
# plt.plot(unsprung_vels, label='Unsprung Vels True')
# plt.legend()
# plt.show()

# plt.plot(sprung_vels_est, label='Sprung Vels Est')
# plt.plot(unsprung_vels_est, label='Unsprung Vels Est')
# plt.legend()
# plt.show()



#{'road_type': rt, 'road_length': rl, 'road_number': n, 'profile': curr_profile,
#                'velocities': velocities, 'gn0': gn0, 'output_directory': output_directory, 'acc_noise': acc_noise,
#                'F_known': F_known, 'Q_known': Q_known, 'fu_mu_known': fu_mu_known, 'sample_rates': sample_rates, 'n_param_inits': n_param_inits,
#                'order': order}

est_eps = ret_dict_avg['Est_Eps_Avg']
est_ws = ret_dict_avg['Est_Ws_Avg']
est_wu = ret_dict_avg['Est_Wu_Avg']
est_xi = ret_dict_avg['Est_Xi_Avg']
est_x0 = ret_dict_avg['Avg_X0']
est_P0 = ret_dict_avg['Avg_P0']
est_Ps = ret_dict_avg['Avg_Ps']
print("est x0 shape: {0}".format(est_x0.shape))
print("est p0.shape: {0}".format(est_P0.shape))
F_final, H, Q, R_prof = emexp.initialize_kf_est_profile(est_eps, est_ws, est_wu, est_xi, road_sigma, est_Ps, acc_noise_level, 1/sample_rate)
#print("F close to: {0}".format())
print("F_final: {0}".format(F_final))
F_true = np.array([[-1*test_car.c, -1*test_car.k2, test_car.c, test_car.k2, 0], 
[1, 0, 0, 0, 0], [test_car.c/test_car.mu, test_car.k2/test_car.mu, -1*test_car.c / test_car.mu, -(test_car.k1 + test_car.k2) / test_car.mu, test_car.k1/test_car.mu],
[0, 0, 1, 0, 0], [0,  0, 0, 0, 0]])
F_close = np.eye(5) + 1/sample_rate * F_true  + (1/sample_rate)**2 * np.matmul(F_true, F_true)/2 + (1/sample_rate)**3 * np.linalg.multi_dot([F_true, F_true, F_true])/6 
print("F_final should be close to: {0}".format(expm(F_true*1/sample_rate)))
print("H_final: {0}".format(H))
print("H final should be: {0}, {1}".format(test_car.c, test_car.k2))
print("Q: {0}".format(Q))
print("R_prof: {0}".format(R_prof))
est_prof, measurements = emexp.estimate_profile(est_eps, est_ws, est_wu, est_xi, road_sigma, est_x0, est_P0,  est_xs, est_Ps, accs_plus_noise.reshape(-1, 1), acc_noise_level, 1/sample_rate)


plt.plot(sprung_vels_est, label='Sprung Vels Est')
plt.plot(measurements[:, 1], label='Sprung Vels Input')
plt.legend()
plt.show()

plt.plot(sprung_disps_est, label='Sprung Disps Est')
plt.plot(measurements[:, 2], label='Sprung Disps Input')
plt.legend()
plt.show()

plt.plot(unsprung_vels_est, label='Unsrung Vels Est')
plt.plot(measurements[:, 3], label='Unsprung Vels Input')
plt.legend()
plt.show()

plt.plot(unsprung_disps_est, label='Unsrung Disps Est')
plt.plot(measurements[:, 4], label='Unsprung Disps Input')
plt.legend()
plt.show()

plt.plot(prof_input, label='Input Profile')
plt.plot(est_prof, label='Est Profile')
plt.legend()
plt.show()