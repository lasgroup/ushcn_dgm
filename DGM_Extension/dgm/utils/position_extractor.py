import numpy as np
import os

PATH_TO_STATIONS_FILE = "/home/philippe/Documents/Code/real_world_data/GRU_ODE/gru_ode_bayes/data_preproc/Climate/ushcn-stations.txt"
PATH_TO_MAPPING = "/home/philippe/Documents/Code/real_world_data/GRU_ODE/gru_ode_bayes/data_preproc/Climate/centers_id_mapping_with_daylength.npy"


class PositionExtractor(object):

    def __init__(self,
                 path_to_stations_file,
                 path_to_mapping=None,
                 ):
        self.position_dict = dict()
        self.elevation_dict = dict()
        with open(os.path.expanduser(path_to_stations_file)) as file:
            while (line := file.readline().rstrip()):
                split_line = line.split()
                curr_coop = str(int(split_line[0]))  # weird construction to remove leading 0s
                curr_lat = float(split_line[1])
                curr_long = float(split_line[2])
                self.position_dict[curr_coop] = [curr_lat, curr_long]
                self.elevation_dict[curr_coop] = [float(split_line[3])]
        if path_to_mapping:
            self.coop_id_to_index_map_dict = np.load(os.path.expanduser(path_to_mapping), allow_pickle=True)
            self.coop_id_to_index_map_dict = self.coop_id_to_index_map_dict.item()
            self.index_to_coop_id_map_dict = dict()
            for key, value in self.coop_id_to_index_map_dict.items():
                self.index_to_coop_id_map_dict[value] = key

    def get_position(self, coop_id: int):
        """
        takes a coop_id as integer input
        returns a list of [latitude, longitude] as output
        """
        return self.position_dict[str(coop_id)]

    def get_elevation(self, coop_id: int):
        """
        takes a coop_id as integer input
        returns a list of [elevation] as output
        """
        return self.elevation_dict[str(coop_id)]

    def get_position_and_elevation_from_index(self, index: int):
        coop_id = self.index_to_coop_id_map_dict[index]
        position = self.get_position(coop_id)
        elevation = self.get_elevation(coop_id)
        return position, elevation

if __name__ == "__main__":
    position_extractor = PositionExtractor(path_to_stations_file=PATH_TO_STATIONS_FILE, path_to_mapping=PATH_TO_MAPPING)
    print(position_extractor.get_position(210018))
    print(position_extractor.get_position(11084))

    print("")
    print(position_extractor.get_position_and_elevation_from_index(0))
    print(position_extractor.get_position_and_elevation_from_index(1))
    print(position_extractor.get_position_and_elevation_from_index(2))
    print(position_extractor.get_position_and_elevation_from_index(3))
    print(position_extractor.get_position_and_elevation_from_index(4))
    print(position_extractor.get_position_and_elevation_from_index(5))
    print(position_extractor.get_position_and_elevation_from_index(6))
