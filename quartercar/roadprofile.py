



class RoadProfile():

    def __init__(self, elevations, distances):
        """

        :param elevations: A list of elevations (in mm)
        :param distances:
        """
        self.elevations = elevations
        self.distances = distances


    def clean(self, distance):
        """
        :param distance: The length in mm desired between road elevation measurements
        :return: A new immutable RoadProfile class instance
        """
        pass

    def length(self):
        """

        :return: The length of the entire profile in meters
        """
        pass

    def split(self, distance):
        """

        :param distance: The distance in meters in which the road profile should be split
        :return: A list of RoadProfile split via distance
        """
        pass

    def to_iri(self):
        """
        :return: The IRI value over the entire road profile
        """
        pass