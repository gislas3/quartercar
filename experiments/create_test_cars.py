import numpy as np



# epsilon (m_s/m_u): avg = 3 to 8, min = 2, max = 20
# omega_s (sqrt(k_s/m_s)): avg = 1, min = .2, max = 1
# omega_u (sqrt(k_u/m_u)): avg = 10, min = 2, max = 20
# xi (c_s/(2*m_s*omega_s): avg = .55, min = 0, max = 2

#From https://www.ridetech.com/info/tech/spring-rate-calculator/:
#Body Type	        Avg Weight(lbs)	    Avg Front Corner(55%)	Avg Rear Corner(45%)
#Street Rod	        2500-4000	        687-1100	                562-900
#Coupe/Convertible	3000-4000	        825-1100	                675-900
#Sedan	            3000-5000	        825-1375	                675-1125
#SUV	            5000-8000	        1375-2200	               1125-1800
#Street Trucks	    5000-8000	        1375-2200	                1125-1800


#Unsprung weight = 1/2 the front or rear weight - IDk what this is referring to, because it also says this:
#Front Unsprung corner weight is usually around 100-125 lbs.


class Car():

    def __init__(self, body_type, total_weight):
        self.body_type = body_type
        self.weight = total_weight



class StreetRod(Car):

    pass

class Coupe_Convertible(Car):
    pass

class Sedan(Car):
    pass

class SUV(Car):
    pass

class Street_Truck(Car):
    pass




