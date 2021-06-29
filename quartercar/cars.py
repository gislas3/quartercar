import numpy as np

from quartercar import qc

# This is a handy program to have all the quarter car parameters I have encountered in one place.

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

def get_car_list():
    """
    Returns a list of all the quartercar parameters I have found so far.
    """
    #car1 = qc.QC(m_s=380, m_u=55, k_s=20000, k_u=350000, c_s=2000)
    #car2 = qc.QC(m_s=380, m_u=55, k_s=60000, k_u=350000, c_s=8000)

    #nissan = qc.QC(epsilon=7.49803834e+00, omega_s=8.20081439e+00, omega_u=1.99376650e+02, xi=6.50572966e-01)
    #nissan = qc.QC(epsilon=6.03192969e+00, omega_s=1.18815121e+01, omega_u=1.87737414e+02, xi=4.95521488e-01)
    #nissan = qc.QC(epsilon=1.37313204e+01, omega_s=1.25172854e+01, omega_u=1.93174379e+02, xi=3.47041919e-01)
    return [('car1', car1), ('car2', car2), ('car3', car3),  ('car4', car4), 
    ('car5', car5), ('car6', car6), ('car7', car7), ('suv1', suv1), ('suv2', suv2), ('bus', bus)]

def get_car_dict():
    """
    Returns a dictionary of all the quartercar parameters I have found so far.
    """
    #nissan = qc.QC(epsilon=7.49803834e+00, omega_s=8.20081439e+00, omega_u=1.99376650e+02, xi=6.50572966e-01)
    #nissan = qc.QC(epsilon=6.03192969e+00, omega_s=1.18815121e+01, omega_u=1.87737414e+02, xi=4.95521488e-01)
    #nissan = qc.QC(epsilon=1.37313204e+01, omega_s=1.25172854e+01, omega_u=1.93174379e+02, xi=3.47041919e-01)
    return {'car1': car1, 'car2': car2, 'car3': car3,  'car4': car4, 
        'car5': car5, 'car6': car6, 'car7': car7, 'suv1': suv1, 'suv2': suv2, 'bus': bus}

def get_random_car(rng=None):
    """
    Returns a random car from the car list
    """
    if rng is None:
        car_tup = np.random.choice(get_car_list())
    else:
        car_tup = rng.choice(get_car_list())
    return car_tup[1]