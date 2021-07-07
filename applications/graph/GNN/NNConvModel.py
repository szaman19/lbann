import lbann
import math
from lbann.util import str_list
from lbann.modules.graph import NNConv
from lbann.modules import ChannelwiseFullyConnectedModule
import numpy as np

def _xavier_uniform_init(fan_in, fan_out):
	""" Xavier uniform initializer

	Args:
		fan_in (int): input size of the learning layer
		fan_out (int): output size of the learning layer
	Returns:
		(UniformInitializer): return an lbann UniformInitializer object
		"""
	a = math.sqrt(6 / (fan_in + fan_out))
	return lbann.UniformInitializer(min=-a, max=a)


def BondEncoder(edge_feature_columns,
			EDGE_EMBEDDING_DIM):
	"""Embeds the edge features into a vector
	Args:
		edge_feature_columns (list(Layers)): A list of layers with edge feaures with shape (NUM_EDGES)
		EDGE_EMBEDDING_DIM (int): The embedding dimensionality of the edge feature vector
	Returns:
		(Layer): A layer containing the embedded edge feature matrix of shape (NUM_EDGES, EDGE_EMBEDDING_DIM)
		"""
    # Courtesy of OGB
	bond_feature_dims = [5, 6, 2]
	_fan_in = bond_feature_dims[0]
	_fan_out = EDGE_EMBEDDING_DIM
	_embedding_weights = lbann.Weights(initializer=_xavier_uniform_init(_fan_in, _fan_out),
		name="bond_encoder_weights_{}".format(0))

	temp = lbann.Embedding(edge_feature_columns[0],
		num_embeddings=bond_feature_dims[0],
		embedding_dim=EDGE_EMBEDDING_DIM,
		weights=_embedding_weights,
		name="Bond_Embedding_0")

	for i in range(1, 3):
		_fan_in = bond_feature_dims[i]
		_fan_out = EDGE_EMBEDDING_DIM
		_embedding_weights = lbann.Weights(initializer=_xavier_uniform_init(_fan_in, _fan_out),
			name="bond_encoder_weights_{}".format(i))
		_temp2 = lbann.Embedding(edge_feature_columns[i],
			num_embeddings=bond_feature_dims[i],
			embedding_dim=EDGE_EMBEDDING_DIM,
			weights=_embedding_weights,
			name="Bond_Embedding_{}".format(i))
		temp = lbann.Sum(temp, _temp2)
	return temp


def AtomEncoder(node_feature_columns,
    		EMBEDDING_DIM):
	"""Embeds the node features into a vector

	Args:
		edge_feature_columns (list(Layers)): A list of layers with node feaures with shape (NUM_NODES)
		EMBEDDING_DIM (int): The embedding dimensionality of the node feature vector
	Returns:
		(Layer): A layer containing the embedded node feature matrix of shape (NUM_NODES, EMBEDDING_DIM)
		"""
    # Courtesy of OGB
	atom_feature_dims = [119, 4, 12, 12, 10, 6, 6, 2, 2]

	_fan_in = atom_feature_dims[0]
	_fan_out = EMBEDDING_DIM

	_embedding_weights = lbann.Weights(initializer=_xavier_uniform_init(_fan_in, _fan_out),
		name="atom_encoder_weights_{}".format(0))

	temp = lbann.Embedding(node_feature_columns[0],
		num_embeddings=atom_feature_dims[0],
		embedding_dim=EMBEDDING_DIM,
		weights=_embedding_weights,
		name="Atom_Embedding_0")
	for i in range(1, 9):
		_fan_in = atom_feature_dims[i]
		_fan_out = EMBEDDING_DIM
		_embedding_weights = lbann.Weights(initializer=_xavier_uniform_init(_fan_in, _fan_out),
			name="atom_encoder_weights_{}".format(i))
		_temp2 = lbann.Embedding(node_feature_columns[i],
			num_embeddings=atom_feature_dims[i],
			embedding_dim=EMBEDDING_DIM,
			weights=_embedding_weights,
			name="Atom_Embedding_{}".format(i))
		temp = lbann.Sum(temp, _temp2)
	return temp


def _index_2d(index, dim1, dim2):
	"""Modified index layer such that it works with value layer with dimension (dim1, dim2)
	 	for use with gather/scatter layers
	 Args:
	 	index (Layer): Layer containig index vector with shape (dim1)
	 	dim1 (int): The number of elements in the index layer 
	 	dim2 (int): The number of features to extend for
	 Returns:
	 	(Layer): New index layer with shape (dim1*dim2)
	 	"""
	offset = lbann.Constant(value=dim2,num_neurons=str_list([dim1, 1]))
	offset_target_indices = lbann.Multiply(index, offset)

	offset_values = np.tile(range(dim2), dim1)

	offset_mat_vals = lbann.Weights(
		initializer=lbann.ValueInitializer(values=str_list(offset_values)),
		optimizer=lbann.NoOptimizer())
	offset_mat = lbann.WeightsLayer(weights=offset_mat_vals, dims=str_list([dim1, dim2]))

	indices = lbann.Tessellate(offset_target_indices, dims=str_list([dim1, dim2]))

	modified_edge_indices = lbann.Reshape(lbann.Sum(offset_mat, indices), dims=str_list([dim1 * dim2]))

	return modified_edge_indices


def graph_data_splitter(_input,
	 					NUM_NODES,
	 					NUM_EDGES,
	 					NUM_NODES_FEATURES,
	 					NUM_EDGE_FEATURES,
	 					EMBEDDING_DIM,
	 					EDGE_EMBEDDING_DIM):
	"""Helper function to split the input data into  

	Args: 
		NUM_NODES (int): The number of nodes in the largest graph in the dataset (51 for LSC-PPQM4M)
      	NUM_EDGES (int): The number of edges in the largest graph in the dataset (118 for LSC-PPQM4M)
      	NUM_NODES_FEATURES (int): The dimensionality of the input node features vector (9 for LSC-PPQM4M)
      	NUM_EDGE_FEATURES (int): The dimensionality of the input edge feature vectors (3 for LSC-PPQM4M)
      	EMBEDDOIN_DIM (int): The embedding dimensionality of the node feature vector
      	EDGE_EMBEDDING_DIM (int): The embedding dimensionality of the edge feature vector 
	Returns:
		(Layer, Layer, Layer, Layer, Layer): Returns 5 Layers. The embedded node feature matrix, the 
											 neighbord nodes feature tensor, the embedded edge feature matrix,
											 the source node index vector, and the label
											 """
	split_indices = []

	start_index = 0
	split_indices.append(start_index)

	node_feature = [NUM_NODES for i in range(1, NUM_NODES_FEATURES + 1)]

	split_indices.extend(node_feature)

	edge_features = [NUM_EDGES for i in range(1, NUM_EDGE_FEATURES + 1)]

	split_indices.extend(edge_features)

	edge_indices_sources = NUM_EDGES
	split_indices.append(edge_indices_sources)

	edge_indices_targets = NUM_EDGES
	split_indices.append(edge_indices_targets)

	# node_mask = NUM_NODES
	# split_indices.append(node_mask)

	target = 1
	split_indices.append(target)

	for i in range(1, len(split_indices)):
 		split_indices[i] = split_indices[i] + split_indices[i - 1]

	graph_input = lbann.Slice(_input, axis=0,
 		slice_points=str_list(split_indices))

	node_feature_columns = [lbann.Reshape(lbann.Identity(graph_input),
		dims=str_list([NUM_NODES]),
		name="node_ft_{}_col".format(x)) for x in range(NUM_NODES_FEATURES)]

	edge_feature_columns = [lbann.Reshape(lbann.Identity(graph_input),
		dims=str_list([NUM_EDGES]),
		name="edge_ft_{}_col".format(x)) for x in range(NUM_EDGE_FEATURES)]

	source_nodes = lbann.Reshape(lbann.Identity(graph_input),
		dims=str_list([NUM_EDGES, 1]),
		name="source_nodes")
	target_nodes = lbann.Reshape(lbann.Identity(graph_input),
		dims=str_list([NUM_EDGES, 1]),
		name="target_nodes")
	# nodes_mask = lbann.Reshape(lbann.Identity(graph_input),
	# 	dims=str_list([NUM_NODES, 1]),
	# 	name="masked_nodes")
	label = lbann.Reshape(lbann.Identity(graph_input),
		dims=str_list([1]),
		name="Graph_Label")

	offset_values = np.tile(range(EMBEDDING_DIM), NUM_EDGES)

	offset_mat_vals = lbann.Weights(
		initializer=lbann.ValueInitializer(values=str_list(offset_values)),
		optimizer=lbann.NoOptimizer())

	offset_mat = lbann.WeightsLayer(weights=offset_mat_vals, dims=str_list([NUM_EDGES, EMBEDDING_DIM]), name="INDEX_OFFSET_MAP")

	indices = lbann.Tessellate(target_nodes, dims=str_list([NUM_EDGES, EMBEDDING_DIM]))

	modified_edge_indices = lbann.Reshape(lbann.Sum(offset_mat, indices), dims=str_list([NUM_EDGES * EMBEDDING_DIM]))

	neighbor_feature_dims = str_list([NUM_EDGES, 1, EMBEDDING_DIM])

	embedded_node_features = AtomEncoder(node_feature_columns, EMBEDDING_DIM)

	embedded_edge_features = BondEncoder(edge_feature_columns, EDGE_EMBEDDING_DIM)

	neighbor_features = lbann.Reshape(embedded_node_features,
		dims=str_list([NUM_NODES * EMBEDDING_DIM]),
		name='Flattened_Neighbor_features')

	neighbor_features = lbann.Gather(neighbor_features, modified_edge_indices)

	neighbor_feature_mat = lbann.Reshape(neighbor_features,
		dims=neighbor_feature_dims)

	return \
	embedded_node_features, neighbor_feature_mat, embedded_edge_features, source_nodes, label


def NNConvLayer(node_features,
				neighbor_features,
				edge_features,
				edge_index,
				in_channel,
				out_channel,
				NUM_NODES,
				NUM_EDGES):
	""" Helper function to create a NNConvLayer with a 3-layer MLP kernel

	Args: node_features (Layer): Layer containing the node featue matrix of the graph (NUM_NODES, in_channel)
	      neighbor_features (Layer): Layer containing the neighbor feature tensor of the graph of shape (NUM_EDGES, 1, in_channel)
	      edge_features (Layer): Layer containing the edge feature matrix of the graph of shape (NUM_EDGES, EMBEDDED_EDGE_FEATURES)
	      edge_index (Layer): Layer contain the source edge index vector of the graph of shape (NUM_EDGES) 
	      in_channel (int): The embedding dimensionality of the node feature vector
	      out_channel (int): The dimensionality of the node feature vectors after graph convolutions
	      NUM_NODES (int): The number of nodes in the largest graph in the dataset (51 for LSC-PPQM4M)
	      NUM_EDGES (int): The number of edges in the largest graph in the dataset (118 for LSC-PPQM4M) 
	      """
	FC = ChannelwiseFullyConnectedModule

	k_1 = math.sqrt(1 / in_channel)
	k_2 = math.sqrt(1 / 64)
	k_3 = math.sqrt(1 / 32)
	nn_sq_1_weight = lbann.Weights(initializer=lbann.UniformInitializer(min=-k_1, max=k_1),
		name="gnn_weights_{}".format(0))
	nn_sq_2_weight = lbann.Weights(initializer=lbann.UniformInitializer(min=-k_2, max=k_2),
		name="gnn_weights_weights_{}".format(1))
	nn_sq_3_weight = lbann.Weights(initializer=lbann.UniformInitializer(min=-k_3, max=k_3),
		name="gnn_weights_weights_{}".format(2))

	sequential_nn = \
	[FC(64, weights=[nn_sq_1_weight], name="NN_SQ_1", bias=True),
	lbann.Relu,
	FC(32, weights=[nn_sq_2_weight], name="NN_SQ_2", bias=True),
	lbann.Relu,
	FC(out_channel * in_channel, weights=[nn_sq_3_weight], name="NN_SQ_3", bias=True)]

	nn_conv = NNConv(sequential_nn,
		NUM_NODES,
		NUM_EDGES,
		in_channel,
		out_channel)

	out = nn_conv(node_features,
		neighbor_features,
		edge_features,
		edge_index)
	return out

def make_model(NUM_NODES,
			   NUM_EDGES,
			   NUM_NODES_FEATURES,
			   NUM_EDGE_FEATURES,
	 		   EMBEDDING_DIM,
			   EDGE_EMBEDDING_DIM,
			   NUM_OUT_FEATURES,
			   NUM_EPOCHS):
	""" Creates an LBANN model for the OGB-LSC PPQM4M Dataset

	Args: 
		NUM_NODES (int): The number of nodes in the largest graph in the dataset (51 for LSC-PPQM4M)
      	NUM_EDGES (int): The number of edges in the largest graph in the dataset (118 for LSC-PPQM4M)
      	NUM_NODES_FEATURES (int): The dimensionality of the input node features vector (9 for LSC-PPQM4M)
      	NUM_EDGE_FEATURES (int): The dimensionality of the input edge feature vectors (3 for LSC-PPQM4M)
      	EMBEDDOIN_DIM (int): The embedding dimensionality of the node feature vector
      	EDGE_EMBEDDING_DIM (int): The embedding dimensionality of the edge feature vector
      	NUM_OUT_FEATURES (int): The dimensionality of the node feature vectors after graph convolutions 
      	NUM_EPOCHS (int): The number of epochs to train the network 
	Returns:
		(Model): lbann model object
		"""
	in_channel = EMBEDDING_DIM
	out_channel = NUM_OUT_FEATURES
	output_dimension = 1

	_input = lbann.Input(target_mode='N/A')
	node_feature_mat, neighbor_feature_mat, edge_feature_mat, edge_indices, target = \
		graph_data_splitter(_input,
							NUM_NODES,
							NUM_EDGES,
							NUM_NODES_FEATURES,
	 						NUM_EDGE_FEATURES,
							EMBEDDING_DIM,
							EDGE_EMBEDDING_DIM)

	modified_edge_indices = _index_2d(edge_indices, NUM_EDGES, out_channel)

	x = NNConvLayer(node_feature_mat,
					neighbor_feature_mat,
					edge_feature_mat,
					modified_edge_indices,
					in_channel,
					out_channel,
					NUM_NODES,
					NUM_EDGES)

	for i, num_neurons in enumerate([256, 128, 32, 8], 1):
		x = lbann.FullyConnected(x,
								num_neurons=num_neurons,
								name="hidden_layer_{}".format(i))
		x = lbann.Relu(x, name='hidden_layer_{}_activation'.format(i))
	x = lbann.FullyConnected(x,
		num_neurons=output_dimension,
		name="output")

	loss = lbann.MeanAbsoluteError(x, target)

	layers = lbann.traverse_layer_graph(_input)
	training_output = lbann.CallbackPrint(interval=1,
		print_global_stat_only=False)
	gpu_usage = lbann.CallbackGPUMemoryUsage()
	timer = lbann.CallbackTimer()

	callbacks = [training_output, gpu_usage, timer]
	model = lbann.Model(NUM_EPOCHS,
		layers=layers,
		objective_function=loss,
		callbacks=callbacks)
	return model
