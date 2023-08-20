from collections import namedtuple
import json


class Config(object):
    """Configuration module."""

    def __init__(self, config):
        self.paths = ""
        # Load config file
        with open(config, 'r') as config:
            self.config = json.load(config)
        # Extract configuration
        self.extract()

    def extract(self):
        config = self.config

        # -- nodes --
        fields = ['total', 'participants_per_round', 'aggregators_per_round', 'bc', 'test_partition', 'selection', 'source']
        defaults = (0,0,0,0, 0.2, 'score', 'generated_nodes.csv')
        params = [config['nodes'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.nodes = namedtuple('nodes', fields)(*params)

        # -- Data --
        fields = ['loading','IID', 'partition']
        defaults = ('static','true', 0)
        params = [config['data'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.data = namedtuple('data', fields)(*params)

        self.loader = 'basic'

        # # Determine correct data loader
        # #assert self.data.IID ^ bool(self.data.bias) ^ bool(self.data.shard)
        # if self.data.IID:
            # self.loader = 'basic'
        # elif self.data.bias:
        #     self.loader = 'bias'
        # elif self.data.shard:
        #     self.loader = 'shard'

        # -- Federated learning --
        fields = ['rounds', 'target_accuracy', 'epochs', 'batch_size', 'x', 'alpha','honesty_alpha','honesty_beta','honesty_gamma','honesty_phi','local_validation_threshold','intermediaire_validation_threshold', 'malus']
        defaults = (0, None, 'train', 0, 0, 0.7,1,5,5,2,10,0.05,10)
        params = [config['federated_learning'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.fl = namedtuple('fl', fields)(*params)

        # -- Model --
        fields = ['name', 'size']
        defaults = ('MNIST', 1600)
        params = [config['model'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.model = namedtuple('model', fields)(*params)

        # -- Paths --
        fields = ['data', 'model','FLmodels', 'reports', 'plot']
        defaults = ('./data', './models', './models.csv', None, './plots')
        params = [config['paths'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        # # Set specific model path
        # params[fields.index('model')] += '/' + self.model.name

        self.paths = namedtuple('paths', fields)(*params)

        # # -- Server --
        # self.server = config['server']

        # # -- Async --
        # fields = ['alpha', 'staleness_func']
        # defaults = (0.9, 'constant')
        # params = [config['async'].get(field, defaults[i])
        #           for i, field in enumerate(fields)]
        # self.sync = namedtuple('sync', fields)(*params)

        # # -- Link Speed --
        # fields = ['min', 'max', 'std']
        # defaults = (200, 5000, 100)
        # params = [config['link_speed'].get(field, defaults[i])
        #           for i, field in enumerate(fields)]
        # self.link = namedtuple('link_speed', fields)(*params)

        # # -- Network Settings --
        # fields = ['type', 'wifi', "ethernet"]
        # defaults = ("wifi", None, None)
        # params = [config['network'].get(field, defaults[i])
        #           for i, field in enumerate(fields)]
        # self.network = namedtuple('network', fields)(*params)

        # -- Plot interval --
        self.plot_interval = config['plot_interval']
