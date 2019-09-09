import numpy
from .roadprofile import RoadProfile

"""
This is a reST style.

:param param1: this is a first param
:param param2: this is a second param
:returns: this is a description of what is returned
:raises keyError: raises an exception
"""



class QC():


    def __init__(self): #TODO: Put in all different car parameters (with defaults) as constructor arguments
        """set car parameters here, e.g. tire spring, suspension spring, damping constant, etc.
        """
        #self.tire_spring =
        pass


    def run(self, road_profile, velocities):
        """
        This is where we generate the acceleration values from the road profile, running the entire QC simulation

        :param road_profile: An array or list like of floating point values (in mm) of road profile elevations
        :param distances: An array that contains the distance (in mm) from start of road profile
        :param velocities: An array that contains the instantaneous velocity of the car at each distance

        :return: list of acceleration values (in m/s^2)


        """
        #
        #return accs
        pass

    def inverse(self, accelerations, distances, velocities):

        """


        :param accelerations: An array of vertical accelerations in (m/s^2) (assumes that preprocessing step,
            subtracting gravity/accounting for orientation has already occurred)
        :param distances: An array that contains the distance (in mm) from start of measurements
        :param velocities: An array that contains the instantaneous velocity of the car at each distance

        :return: `RoadProfile`
        """

        #return road_profile
        pass






