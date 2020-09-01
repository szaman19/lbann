import numpy as np 


class Synthetic_Data:
    def __init__(self, num_samples = 1000):
        self.num_samples = num_samples
        
        self.data = np.random(num_samples, 19, 48,48,48)

    def __len__(self):
        return self.num_samples 
    
    def __getitem__(self, idx):

        return np.concatenate((np.flatten(np.float32(self.data[idx])),
        		np.float32(np.random.normal())), axis=None)        


class Synthetic_Graph_Data(object):
	"""docstring for Synthetic_Graph_Data"""
	def __init__(self, num_samples = 100 , num_nodes = 100):
		super(Synthetic_Graph_Data, self).__init__()
		self.num_samples = num_samples
		self.num_nodes = 100
		self.adj_matrices = []
		self.covalent_nodes = []
		self.non_covalent_nodes = []

	def __len__(self):
		return self.num_samples

	def __getitem__(self, idx):
		data = np.concatenate((self.adj_matrices[idx].flatten(),
							   self.node_features[idx].flatten(),
							   np.float32(np.random.normal())), 
							  axis=None)
		return data
		