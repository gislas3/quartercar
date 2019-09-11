import numpy
from .roadprofile import RoadProfile

class QC():
    """
    A quarter car model that represents the interplay between a road surface (`RoadProfile`), a fixed (and always grounded) mass and a sprung mass.
    The class 
      1) calculates acceleration values from a `RoadProfile` and velocities (see `run`) and 
      2) reverses the calculation to generate a `RoadProfile` from accelerations, distances and velocities.
    """

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






