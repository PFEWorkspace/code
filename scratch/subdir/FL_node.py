

class Node(object):

    def __init__(self, nodeinfo) -> None:
        self.node = nodeinfo
        self.loss = 10.0 # set intial loss big 
    
    def download(self, argv):
        try:
            return argv.copy()
        except:
            return argv

    def upload(self, argv):
        try:
            return argv.copy()
        except:
            return argv


    def set_data(self, data, config):
        # Extract from config
        test_partition = self.test_partition = config.nodes.test_partition

        # Download data
        self.data = self.download(data)

        # Extract trainset, testset (if applicable)
        self.trainset = data[:int(len(self.data) * (1 - test_partition))]
        self.testset = data[int(len(self.data) * (1 - test_partition)):]
       
    