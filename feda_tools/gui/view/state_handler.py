import yaml

class StateHandler:
    def __init__(self, config_file='config.yaml'):
        self.file_ptu = None
        self.file_irf = None
        self.file_bkg = None
        self.output_directory = None
        self.chunk_size = None
        self.threshold_multiplier = None
        self.data_ptu = None
        self.data_irf = None
        self.data_bkg = None
        self.burst_index = None
        self.fit_params = None
        self.process_results = None
        self.mu = None
        self.std = None
        self.config = self.load_config(config_file)

    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
