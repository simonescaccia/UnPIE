from abc import abstractmethod


class Data(object):
    # abstract method generate_data_sequence
    @abstractmethod
    def generate_data_sequence(self, **kwargs):
        pass