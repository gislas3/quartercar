from quartercar import qc
from quartercar import roadprofile
from tests import make_profile
import numpy as np
import pandas as pd
import argparse



def generate_road_profiles(num_roads, sigma_seed=15, ):
    # Step 1: Generate 100 road profiles of 100 meters each
    #sigma_seed = 15
    np.random.seed(sigma_seed)

    # smooth road profiles:
    sigmas1 = np.abs(np.random.normal(6, 3, 33))

    # rougher road profiles
    sigmas2 = np.abs(np.random.normal(15, 2, 33))

    # even rougher road profiles
    sigmas3 = np.abs(np.random.normal(25, 2, 34))

    road_profiles = {}

    profile_len, delta, cutoff_freq, delta2 = 100, .1, .15, .01

    for x in range(0, 100):
        #seed = x
        if (x <= 32):
            sigma = sigmas1[x]
        elif (x > 32 and x <= 65):
            sigma = sigmas2[x - 33]
        else:
            sigma = sigmas3[x - 66]
        dists, orig_hts, low_pass_hts, final_dists, final_heights = make_profile.make_gaussian(sigma, profile_len, delta,
                                                                                     cutoff_freq, delta2, x)
        profile = roadprofile.RoadProfile(final_dists, final_heights)
        # this is to double check IRIs are properly calculated
        # df1 = pd.DataFrame({"dists": profile.get_distances(), "elevations": profile.get_elevations()})
        # df1.to_csv("/Users/gregoryislas/Documents/Mobilized/Test_Profiles/{0}.csv".format(x), index=False)
        # df1 = None

        road_profiles[x] = profile
    return road_profiles

    # iris1, iris2, iris3 = [], [], []
    # here we go...
    # use usual parameters for QC model:


def get_iris(profiles, prefix_str, qc1, velocity, sample_rate_hz, m_s, m_u, c_s, k_s, k_u):
    to_ret_str = ""
    for y in range(0, len(profiles)):
        qc2 = qc.QC(m_s, m_u, c_s, k_s, k_u)
        prof = profiles[y]
        true_iri = prof.to_iri()
        T, yout, xout, new_distances, new_elevations = qc2.run(prof, 100, velocity, sample_rate_hz)
        prof2 = roadprofile.RoadProfile(new_distances, new_elevations, True) #Should be filtered since need to account for enveloping effect of tire
        iri_est_with_params = prof2.to_iri()
        elevations, x_s_dot, x_s, x_u, x_u_dot, x_u_dot_dot = qc1.inverse(accelerations=yout[:, -1], distances=new_distances,
                                                                          velocities=velocity,
                                                                          sample_rate_hz=sample_rate_hz)
        prof3 = roadprofile.RoadProfile(new_distances, elevations, True) #Should be filtered since need to account for enveloping effect of tire
        iri_est_wrong_params = prof3.to_iri()
        to_ret_str += "{0},{1},{2},{3}\n".format(prefix_str, true_iri, iri_est_with_params, iri_est_wrong_params)
    return to_ret_str

def generate_simulations(param, velocity, num_roads):
    profiles = generate_road_profiles(num_roads)
    m_s, m_u, c_s, k_s, k_u = 208, 28, 1300, 18709, 127200  # QC parameters
    qc1 = qc.QC(m_s, m_u, c_s, k_s, k_u)
    #Want to return: True IRI, Computed IRI with True parameters, IRI with incorrect parameter, velocity
    #Here are the ratios for lbs and lbs/inch spring rates:
    # epsilon (m_s/m_u): avg = 3 to 8, min = 2, max = 20
    # omega_s (sqrt(k_s/m_s)): avg = 1, min = .2, max = 1
    # omega_u (sqrt(k_u/m_u)): avg = 10, min = 2, max = 20
    # xi (c_s/(2*m_s*omega_s): avg = .55, min = 0, max = 2
    #Converting to kg and N/m:
    # epsilon (m_s/m_u): avg = 3 to 8, min = 2, max = 20 (same since denominator and numerator are off by same constant)
    # omega_s (sqrt(k_s/m_s)): conversion factor = sqrt((1/175.126835)/2.20462)
    # omega_u (sqrt(k_s/m_s)): conversion factor = sqrt((1/175.126835)/2.20462)
    # xi : No conversion factor needed

    k_factor = np.sqrt((1/175.126835)/2.20462)
    sample_rate_hz = 100 #Since our sensors are sampling at 100 HZ
    tot_str = ""
    if param == 'k_s': #Test variation in suspension spring rate - sqrt(k
        filename = "varying_k_s.csv"
        header = "True_K_S,True_K_S_Ratio,Input_K_S,Input_K_S_Ratio,True_IRI,Est_IRI,Est_IRI_Wrong_Parameter\n"
        for x in np.arange(.2, 1.1, .1):
            k_s_new = m_s * (x / k_factor) ** 2  # convert to m_s
            prefix = "{0},{1},{2},{3}".format(np.round(k_s_new, 3), np.round(k_factor*np.sqrt(k_s_new/m_s), 3), k_s, np.round(k_factor*np.sqrt(k_s/m_s), 3))
            tot_str += get_iris(profiles, prefix, qc1, velocity, sample_rate_hz, m_s, m_u, c_s, k_s_new, k_u)

    elif param == 'k_u': #Test variation in tire spring rate
        filename = "varying_k_u.csv"
        header = "True_K_U,True_K_U_Ratio,Input_K_U,Input_K_U_Ratio,True_IRI,Est_IRI,Est_IRI_Wrong_Parameter\n"
        for x in np.arange(2, 21, 1):
            k_u_new = m_u * (x / k_factor) ** 2  # convert to m_s
            prefix = "{0},{1},{2},{3}".format(np.round(k_u_new, 3), np.round(k_factor * np.sqrt(k_u_new / m_u), 3), k_u,
                                              np.round(k_factor * np.sqrt(k_u / m_u), 3))
            tot_str += get_iris(profiles, prefix, qc1, velocity, sample_rate_hz, m_s, m_u, c_s, k_s, k_u_new)

    elif param == 'm_s': #Test variation in sprung mass
        filename = "varying_m_s.csv"
        header = "True_M_S,True_M_S_M_U_Ratio,Input_M_S,Input_M_S_M_U_Ratio,True_IRI,Est_IRI,Est_IRI_Wrong_Parameter\n"
        for x in np.arange(3, 15.5, .5):
            m_s_new = m_u * x
            prefix = "{0},{1},{2},{3}".format(np.round(m_s_new, 3), np.round(m_s_new/m_u, 3), m_s,
                                              np.round(m_s/m_u, 3))
            tot_str += get_iris(profiles, prefix, qc1, velocity, sample_rate_hz, m_s_new, m_u, c_s, k_s, k_u)

    elif param == 'm_u': #Test variation in unsprung mass
        filename = "varying_m_u.csv"
        header = "True_M_U,True_M_S_M_U_Ratio,Input_M_U,Input_M_S_M_U_Ratio,True_IRI,Est_IRI,Est_IRI_Wrong_Parameter\n"
        for x in np.arange(3, 15.5, .5):
            m_u_new = m_s/x
            prefix = "{0},{1},{2},{3}".format(np.round(m_u_new, 3), np.round(m_s / m_u_new, 3), m_u,
                                              np.round(m_s / m_u, 3))
            tot_str += get_iris(profiles, prefix, qc1, velocity, sample_rate_hz, m_s, m_u_new, c_s, k_s, k_u)

    elif param == 'c_s': #Test variation in damping rate
        filename = "varying_c_s.csv"
        header = "True_C_S,True_Xi,Input_C_X,Input_Xi,True_IRI,Est_IRI,Est_IRI_Wrong_Parameter\n"
        for x in np.arange(.1, 2.2, .2):
            c_s_new = 2*m_s*np.sqrt(k_s/m_s)*x
            prefix = "{0},{1},{2},{3}".format(np.round(c_s_new, 3), np.round(c_s_new /(2*m_s*np.sqrt(k_s/m_s)), 3), c_s,
                                              np.round(c_s/ (2 * m_s * np.sqrt(k_s / m_s)), 3))
            tot_str += get_iris(profiles, prefix, qc1, velocity, sample_rate_hz, m_s, m_u, c_s_new, k_s, k_u)
    else:
        return None
    with open("data/IRI_Var_Params2/{0}".format(filename), "w") as f:
        f.write(header + tot_str)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computes the error for IRI values when one of the car parameters is wrong')
    parser.add_argument('--param', nargs='*', metavar='PARAM',
                        help='The parameter to test')
    parser.add_argument('-v', '--vel', '--velocity', nargs='?', type=int, default=15, const=15, metavar='XX',
                        help='The velocity (in m/s) to use for the car simulations')
    parser.add_argument('-n', '--num_roads', '--num_profiles', nargs='?', type=int, default=100, const=100, metavar='XX',
                        help='The number of road profiles to use for the simulation')

    # parser.add_argument('-r', '--receive_length', '--rml', nargs='?', type=int, default=1024, const=1024, metavar='N',
    #                    help='The number of bytes the server expects to be sent with each send')
    # parser.add_argument('-s', '--send_length', '--sml', nargs='?', type=int, default=4096, const=4096, metavar='N',
    #                    help='The number of bytes the server will send back to the client')
    args = parser.parse_args()
    print("Args is {0}".format(args))
    for param in args.param:
        generate_simulations(param, args.vel, args.num_roads)





