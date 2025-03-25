from abc import abstractmethod


class Data(object):
    def __init__(self):
        self.ped_type = 'peds'
        self.traffic_type = 'objs'

    # abstract method generate_data_sequence
    @abstractmethod
    def generate_data_sequence(self, **kwargs):
        pass

    def get_ped_type(self):
        return self.ped_type
    
    def get_traffic_type(self):
        return self.traffic_type