import lbann
import lbann.contrib.launcher
import lbann.contrib.args

import argparse
import os 
import configparser

import data.LSC_PPQM4M
from lbann.util import str_list
from lbann.modules.graph import NNConv
from lbann.modules import ChannelwiseFullyConnectedModule


desc = ("Training Edge-conditioned Graph Convolutional Model Using LBANN ")

parser = argparse.ArgumentParser(description=desc)

lbann.contrib.args.add_scheduler_arguments(parser)
lbann.contrib.args.add_optimizer_arguments(parser)

parser.add_argument(
    '--num-epochs', action='store', default=100, type=int,
    help='number of epochs (deafult: 100)', metavar='NUM')

parser.add_argument(
    '--mini-batch-size', action='store', default=32, type=int,
    help="mini-batch size (default: 32)", metavar='NUM')

parser.add_argument(
    '--num-edges', action='store', default=118, type=int,
    help='number of edges (deafult: 118)', metavar='NUM')

parser.add_argument(
    '--num-nodes', action='store', default=51, type=int,
    help='number of nodes (deafult: 51)', metavar='NUM')

parser.add_argument(
    '--num-node-features', action='store', default=100, type=int,
    help='number of node features (deafult: 100)', metavar='NUM')

parser.add_argument(
    '--num-edge-features', action='store', default=16, type=int,
    help='number of edge features (deafult: 16)', metavar='NUM')

parser.add_argument(
    '--num-out-features', action='store', default=16, type=int,
    help='number of node features for NNConv (deafult: 16)', metavar='NUM')

parser.add_argument(
    '--num-samples', action='store', default=3045360, type=int,
    help='number of Samples (deafult: 3045360)', metavar='NUM')

parser.add_argument(
    '--job-name', action='store', default="NN_Conv", type=str,
    help="Job name for scheduler", metavar='NAME')

args = parser.parse_args()

kwargs = lbann.contrib.args.get_scheduler_kwargs(args)

MINI_BATCH_SIZE = args.mini_batch_size
NUM_EPOCHS = args.num_epochs
JOB_NAME = args.job_name
NUM_NODES = 51
NUM_EDGES = 118
NUM_NODES_FEATURES = 100
NUM_EDGE_FEATURES = 16
NUM_OUT_FEATURES = args.num_out_features
NUM_SAMPLES = args.num_samples

#----------------------------------------

# Generating configuration for dataset

#----------------------------------------

config = configparser.ConfigParser()
config['Graph'] = {}
config['Graph']['num_nodes'] =  str(NUM_NODES)
config['Graph']['num_edges'] = str(NUM_EDGES)
config['Graph']['num_node_features'] = str(NUM_NODES_FEATURES)
config['Graph']['num_edge_features'] = str(NUM_EDGE_FEATURES)
config['Graph']['num_samples'] = str(NUM_SAMPLES)

current_file = os.path.realpath(__file__)
app_dir = os.path.dirname(current_file)
_file_name = os.path.join(app_dir, 'config.ini')

with open(_file_name, 'w') as configfile:
    config.write(configfile)

os.environ['LBANN_LSC_CONFIG_FILE'] = _file_name

def graph_data_splitter(_input):

    split_indices = []

    start_index = 0
    split_indices.append(start_index)

    node_feature = NUM_NODES * NUM_NODES_FEATURES
    split_indices.append(node_feature)

    edge_features = NUM_EDGES * NUM_EDGE_FEATURES
    split_indices.append(edge_features)
    
    edge_indices_sources = NUM_EDGES
    split_indices.append(edge_indices_sources)

    edge_indices_targets = NUM_EDGES
    split_indices.append(edge_indices_targets)


    target = 1
    split_indices.append(target)
    
    for i in range(1, len(split_indices)):
        split_indices[i] = split_indices[i] + split_indices[i-1]        

    graph_input = lbann.Slice(_input, axis=0,
                              slice_points=str_list(split_indices))
    graph_data_id = [lbann.Identity(graph_input) for x in range(5)]

    node_feature_dims = str_list([NUM_NODES, NUM_NODES_FEATURES])
    
    edge_feature_dims = str_list([NUM_EDGES, NUM_EDGE_FEATURES])
    edge_indices_dims = str_list([NUM_EDGES])
    target_dims = str_list([1])

    node_feature_mat = lbann.Reshape(graph_data_id[0],
                                     dims=node_feature_dims,
                                     name="Input_node_fts_reshape")
    
    edge_feature_mat = lbann.Reshape(graph_data_id[1],
                                     dims=edge_feature_dims,
                                     name="Input_edge_fts_reshape")
    
    edge_indices_targets = lbann.Reshape(graph_data_id[2],
                                 dims=edge_indices_dims,
                                     name="Input_target_indices_reshape")
    edge_indices_sources = lbann.Reshape(graph_data_id[3],
                                 dims=edge_indices_dims,
                                  name="Input_source_indices_reshaped")
    
    target = lbann.Reshape(graph_data_id[4],
                           dims=target_dims)
    
    
    modified_edge_target_indices = []

    for i in range(NUM_NODES_FEATURES):
        offset = lbann.Constant(value=i,
                                num_neurons=str(NUM_EDGES),
                                name="edge_val_target_col_{}".format(i))

        _updated_edge_ind = lbann.Sum(edge_indices_targets, offset)
        modified_edge_target_indices.append(_updated_edge_ind)
    
    modified_edge_indices = lbann.Concatenation(modified_edge_target_indices)
    neighbor_feature_dims = str_list([NUM_EDGES, 1, NUM_NODES_FEATURES])
    
    neighbor_features = lbann.Gather(graph_data_id[0], modified_edge_indices)

    neighbor_feature_mat = lbann.Reshape(neighbor_features,
                                         dims=neighbor_feature_dims)

    return \
        node_feature_mat, neighbor_feature_mat, edge_feature_mat, edge_indices_sources, target


def reduction(graph_feature_matrix, channels):
    vector_size = str_list([1, NUM_NODES])
    reduction_vector = lbann.Constant(value=1,
                                      num_neurons=vector_size,
                                      name='Sum_Vector')
    reduced_features = lbann.MatMul(reduction_vector, graph_feature_matrix,
                                    name='Node_Feature_Reduction')
    reduced_features = lbann.Reshape(reduced_features,
                                     dims=str_list([channels]))
    return reduced_features


def NNConvLayer(node_features,
                neighbor_features,
                edge_features,
                edge_index,
                in_channel,
                out_channel):

    FC = ChannelwiseFullyConnectedModule
    sequential_nn = \
        [FC(1024, name="NN_SQ_1"),
         lbann.Relu,
         FC(512, name="NN_SQ_2"),
         lbann.Relu,
         FC(256, name="NN_SQ_3"),
         lbann.Relu,
         FC(out_channel * in_channel),
         lbann.Relu]

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


def make_model():
    in_channel = NUM_NODES_FEATURES
    out_channel = NUM_OUT_FEATURES
    output_dimension = 1

    _input = lbann.Input(target_mode='N/A')
    node_feature_mat, neighbor_feature_mat, edge_feature_mat, edge_indices, target = \
        graph_data_splitter(_input)
    modified_edge_indices = []
    for i in range(out_channel):
        offset = lbann.Constant(value=i,
                                num_neurons=str(NUM_EDGES),
                                name="edge_val_col_{}".format(i))

        _updated_edge_ind = lbann.Sum(edge_indices, offset)
        modified_edge_indices.append(_updated_edge_ind)
    modified_edge_indices = lbann.Concatenation(modified_edge_indices)
    node_fts = NNConvLayer(node_feature_mat,
                           neighbor_feature_mat,
                           edge_feature_mat,
                           modified_edge_indices,
                           in_channel,
                           out_channel)
    graph_embedding = reduction(node_fts, out_channel)

    x = lbann.FullyConnected(graph_embedding,
                             num_neurons=8,
                             name='hidden_layer_1')
    x = lbann.Relu(x, name='hidden_layer_1_activation')
    x = lbann.FullyConnected(x,
                             num_neurons=output_dimension,
                             name="output")
    loss = lbann.MeanSquaredError(x, target)

    layers = lbann.traverse_layer_graph(_input)
    print_model = lbann.CallbackPrintModelDescription()
    training_output = lbann.CallbackPrint(interval=1,
                                          print_global_stat_only=False)
    gpu_usage = lbann.CallbackGPUMemoryUsage()
    timer = lbann.CallbackTimer()
    callbacks = [print_model, training_output, gpu_usage, timer]
    model = lbann.Model(NUM_EPOCHS,
                        layers=layers,
                        objective_function=loss,
                        callbacks=callbacks)
    return model


model = make_model()
optimizer = lbann.SGD(learn_rate=1e-3)
data_reader = data.LSC_PPQM4M.make_data_reader("LSC_DATA")
trainer = lbann.Trainer(mini_batch_size=MINI_BATCH_SIZE)


lbann.contrib.launcher.run(trainer,
                           model,
                           data_reader,
                           optimizer,
                           job_name=JOB_NAME,
                           **kwargs)

