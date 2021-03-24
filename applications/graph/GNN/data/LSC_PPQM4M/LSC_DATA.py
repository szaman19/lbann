import configparser
import numpy as np
import threading
import os
import os.path
import math
import sys
import warnings





# ----------------------------------------------
# Configuration
# ----------------------------------------------

# Load config file
config_file = os.getenv('LBANN_LSC_CONFIG_FILE')
if not config_file:
    raise RuntimeError(
        'No configuration file provided in '
        'LBANN_LSC_CONFIG_FILE environment variable')
if not os.path.exists(config_file):
    raise FileNotFoundError(f'Could not find config file at {config_file}')


config = configparser.ConfigParser()
config.read(config_file)

# ----------------------------------------------
# MPI Process Info
# ----------------------------------------------

def mpi_rank():
    """Current process's rank within MPI world communcator."""
    if 'OMPI_COMM_WORLD_RANK' not in os.environ:
        warnings.warn(
            'Could not detect MPI environment variables. '
            'We expect that LBANN is run with '
            'Open MPI or one of its derivatives.',
            warning.RuntimeWarning,
        )
    return int(os.getenv('OMPI_COMM_WORLD_RANK', default=0))

def mpi_size():
    """Number of ranks in MPI world communcator."""
    if 'OMPI_COMM_WORLD_RANK' not in os.environ:
        warnings.warn(
            'Could not detect MPI environment variables. '
            'We expect that LBANN is run with '
            'Open MPI or one of its derivatives.',
            warning.RuntimeWarning,
        )
    return int(os.getenv('OMPI_COMM_WORLD_SIZE', default=1))

num_nodes = config.getint('Graph', 'num_nodes')
num_edges = config.getint('Graph', 'num_edges')
num_node_features = config.getint('Graph', 'num_node_features')
num_edge_features = config.getint('Graph', 'num_edge_features')
num_samples = config.getint('Graph', 'num_samples')

# Set numpy random seed
np.random.seed(0) 

def get_indices(world_size, rank):
	"""The set of indices for each rank """
	all_indices = np.arange(num_samples)
	split_indices  = np.array_split(all_indices, world_size)
	return split_indices[rank]



# ----------------------------------------------
# Iterator Class
# ----------------------------------------------

class LSC_PPQM4M_Iterator(object):
	"""docstring for LSC_PPQM4M
	"""
	def __init__(self,
				 indices):
		super(LSC_PPQM4M_Iterator, self).__init__()

		

		self.samples_per_rank = len(indices)

		# print(self.datasets[0].shape)
		self.num_samples = 3045360
		self.num_samples_per_file = 304536 

		self.indices = indices

		# Batch set up 
		self.batch_size = 10000
		self.num_batches = math.ceil(self.samples_per_rank / self.batch_size) 
		self.batch_indices = iter(np.array_split(indices, self.num_batches))

		# Cached batch 
		self.batch = np.zeros((self.batch_size, sample_dims_func()[0]), dtype=np.float32)
		self.batch_sample_index = []
		
		# Next cached batch 

		self.next_batch_thread = None 
		self.next_batch = None

	def __iter__(self):
		return self 
	
	def __next__(self):
		if not self.batch_sample_index: # Check if no more samples left 
			self._generate_batch(batch=self.batch,
								 indices=self.batch_sample_index)
		# print(len(self.batch_sample_index))	
		return self.batch[self.batch_sample_index.pop()]
	
	def _generate_batch(self,
					   	batch,
					   	indices):
		
		if self.next_batch_thread is None:
			self._load_data()
		else:
			self.next_batch_thread.join() # Join the thread
		
		next_batch = self.next_batch.copy() 
		
		self.next_batch_thread = threading.Thread(target=self._load_data)
		self.next_batch_thread.start()

		batch_size = min(self.batch_size, next_batch.shape[0])
		# print(batch_size)
		batch[:batch_size,:] = next_batch 
	
		indices.clear()
		indices.extend(range(batch_size))
		np.random.shuffle(indices)

	def _load_data(self):
		'''
		Load next batch of data in a background thread 
		'''

		try:
			next_batch_indices = next(self.batch_indices)

		except StopIteration: # Finished going through all the sample in one epoch
			self.batch_indices = iter(np.array_split(self.indices, self.num_batches))
			next_batch_indices = next(self.batch_indices)

		# print("Loading Next batch")
		_file_dir = '/p/vast1/zaman2/lbann_mol_graphs_training_{}.bin'
		datasets = [np.memmap(_file_dir.format(x),
							  dtype='float32',
							  shape=(self.num_samples_per_file, sample_dims_func()[0]), mode='r') 
							  for x in range(10)]
		
		_batch_size = len(next_batch_indices)
		_next_batch = np.zeros((_batch_size, sample_dims_func()[0]), dtype=np.float32)
		
		for i, row in enumerate(next_batch_indices):
			_file = row // self.num_samples_per_file
			_index = row - (_file * self.num_samples_per_file)

			_next_batch[i,:] = np.array(datasets[_file][_index])

		self.next_batch = _next_batch


# ----------------------------------------------
# Sample access functions
# ----------------------------------------------

_dataset = None

def num_samples_func():
	return num_samples

def get_sample_func(*args):
	global _dataset

	if _dataset is None:
		world_size = mpi_size()
		world_rank = mpi_rank()
		# print("World Size: ", world_size, "World Rank: ", world_rank)

		_indices = get_indices(world_size, world_rank)
		_dataset = LSC_PPQM4M_Iterator(_indices )

	return next(_dataset) 

def sample_dims_func():
	node_features = num_nodes * num_node_features
	edge_features = num_edges * num_edge_features
	edge_indices = num_edges * 2 
	target = 1 
	return (node_features + edge_features + edge_indices + target, )


if __name__ == '__main__':	
	# _file_dir = '/usr/workspace/wsa/zaman2/LSC_PCQM4M/temp_1.bin'

	# # out = np.memmap(_file_dir, dtype='float32', shape=(3045360, 7225), mode='w+')

	x = np.zeros(sample_dims_func())

	for i in range(num_samples_func()):
		x[:] = get_sample_func(i)

	