import numpy as np

#NOTES:

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


#Unsprung weight = 1/2 the front or rear weight


#May be of use if want to make web crawler to get all this data or just manually copy and paste:
# https://www.ridetech.com/info/tech/vehicle-weights/

#Need to investigate this:
#https://vpic.nhtsa.dot.gov/api/Home
#Curb weight: 54
#Decode VIN - can get curb weight, bus type (no suspension)
#Decode WMI - maybe useful...
#Get WMI's for manufacturer
#Get all makes
#Get all manufacturers
#Get Makes for Manufacturer by Manufacturer Name
#Get Makes for Manufacturer by Manufacturer Name and Year
#Get Makes for Vehicle Type by Vehicle Type Name
#Get Vehicle Types for Make by Name
#Get Equipment Plant Codes
#Get Models for Make and a combination of Year and Vehicle Type
#

#Useful vehicle parameters:
#1. Curb weight
#2. Wheel base type
#3. Wheel base inches
#4. Wheel base inches up to
#5. Note
#6. Number of wheels
#7. Wheel size Front (inches)
#8. Wheel size rear (inches)
#9. Suggested VIN
#10. Axle Configuration
#11. Bus length
#12. Bus Floor Configuration Type
#13. Bus Type
#14. Other Bus Info

#VIN Format:
#1. 1-3: WMI (or 1-2, 3=9); 4-8 Vehicle Attributes; 9 Check Digit; 10 Model Year; 11 Plant Code; 12-17 Sequential Number
#1-3 - Use Get WMI;
#5UXWX7C5*BA