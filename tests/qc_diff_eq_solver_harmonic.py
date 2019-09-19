import numpy as np
from matplotlib import pyplot as plt

def harmonic_solver(m_s, m_u, c_s, k_s, k_u, amplitude, wavelen, velocity, T):
    """
    Given a harmonic exciting force as input to the system of QC differential equations,
    solves for the coefficients of the displacment of the sprung and unsprung masses.
    TODO: Need to be able to solve for arbitrary linear combination of exciting harmonic forces
    :param m_s: The sprung mass (in kg)
    :param m_u: The unsprung mass (in kg)
    :param c_s: The damping coefficient
    :param k_s: The stiffness of the suspension spring
    :param k_u: The stiffneess of the tire spring
    :param amplitude: The apmlitude of the exciting force (i.e. road profile) in meters
    :param wavelen: The wavelength (in meters) of the exciting force
    :param velocity: The velocity the car traveled over (this is necessary to get the proper spacing/frequency in the time domain)
    :param T: The times to output the sprung mass acceleration at
    :return: A list of acceleration values (in units m/s^2) at the request time values for the motion of the sprung mass
    """

    #Attempt at explanataion for everything:
    #System of equations: (x_dot_dot, x_dot are the second and first derivatives, respectively, with respect to time)
    #1. m_s*x_s_dot_dot + c_s(x_s_dot - x_u_dot) + k_s(x_s - x_u) = 0
    #2. m_u*x_u_dot_dot + c_s(x_u_dot - x_s_dot) + (k_u + k_s)*x_u - k_s*x_s = k_u*amplitude*sin(2*pi/wavelen)*t

    ####So basically, you get a profile in the spatial domain, and this returns a solution (for now, just the
    ###acceleration of the sprung mass) in the time domain

    #First step, solve for omega in the time domain
    wavelen_time = wavelen/velocity

    omega_time = 2*np.pi/wavelen_time

    #Particular solution (don't care about general solution since it should diminish with time) will be:
    #x_s = A_1*sin(omega_time*t) + B_1*cos(omega_time*t)
    #x_u = A_2*sin(omega_time*t) + B_2*cos(omega_time*t)

    #Taking the derviatives with the above solutions yields:
    #1. m_s*[-A_1*omega_time**2*sin(omega_time*t) - B_1*omega_time**2*cos(omega_time*t)] +
    #   c_s([A_1*omega_time*cos(omega_time*t) - B_1*omega_time*sin(omega_time*t)] - [A_2*omega_time*cos(omega_time*t) - B_2*omega_time*sin(omega_time*t)])
    # + k_s (A_1*sin(omega_time*t) + B_1*cos(omega_time*t) - [A_2*sin(omega_time*t) + B_2*cos(omega_time*t)])) = 0

    #2. m_u*(-A_2*omega_time**2*sin(omega_time*t) - B_2*omega_time**2*cos(omega_time*t))  +
    # c_s*([A_2*omega_time*cos(omega_time*t) - B_2*omega_time*sin(omega_time*t)] - [A_1*omega_time*cos(omega_time*t) - B_1*omega_time*sin(omega_time*t)])
    # - (k_u + k_s)*(A_2*sin(omega_time*t) + B_2*cos(omega_time*t)) - k_s*(A_1*sin(omega_time*t) + B_1*cos(omega_time*t)) = k_u*amplitude*sin(omega_time*t)

    #Thus, we have the following equations:

    #1. m_s*(-A_1*omega_time**2) + c_s(-B_1*omega_time + B_2*omega_time) + k_s*(A_1 - A_2) = 0
    #2. m_s*(-B_1*omega_time**2) + c_s(A_1*omega_time - A_2*omega_time) + k_s(B_1 - B_2) = 0
    #3. m_u*(-A_2*omega_time**2) + c_s(-B_2*omega_time + B_1*omega_time) + (k_u+k_s)*(A_2) - k_s*A_1 = k_u*amplitude
    #4. m_u*(-B_2*omega_time**2) + c_s(A_2*omega_time - A_1*omega_time) + (k_u + k_s)*B_2 - k_s*B_1 = 0

    #Coefficient order: A_1, A_2, B_1, B_2
    eq_1 = [-m_s*omega_time**2 + k_s, -k_s, -c_s*omega_time, c_s*omega_time]
    eq_2 = [c_s*omega_time, -c_s*omega_time, -m_s*omega_time**2 + k_s, -k_s]
    eq_3 = [-k_s, -m_u*omega_time**2 + k_u+k_s, c_s*omega_time, -c_s*omega_time]
    eq_4 = [-c_s*omega_time, c_s*omega_time, -k_s, -m_u*omega_time**2 + k_u + k_s]

    a = np.array([eq_1, eq_2, eq_3, eq_4])
    #print("A.shape is {0}".format(a.shape))
    b = np.array([0, 0, k_u*amplitude, 0]).reshape(4, -1)
    #print("B.shape is {0}".format(b.shape))
    x = np.linalg.solve(a, b)
    #print(x)
    #xs = np.linspace(0, 10, 1000)
    #TODO: Modify as need be to be able to handle cosine and sine - for now we're assuming whomever is using this function will transform cosine to sine as need be
    ys = -1*x[0]*(omega_time**2)*np.sin(omega_time*T) - x[2]*(omega_time**2)*np.cos(omega_time*T)
    #plt.plot(xs, ys, color='g'
    #plt.show()
    return ys






