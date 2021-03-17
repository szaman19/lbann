import lbann
import lbann.contrib.launcher
import lbann.contrib.args

import argparse

import data.Synthetic
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
    '--num-edges', action='store', default=100, type=int,
    help='number of edges (deafult: 100)', metavar='NUM')

parser.add_argument(
    '--num-nodes', action='store', default=100, type=int,
    help='number of nodes (deafult: 10)', metavar='NUM')

parser.add_argument(
    '--num-node-features', action='store', default=20, type=int,
    help='number of node features (deafult: 20)', metavar='NUM')

parser.add_argument(
    '--num-edge-features', action='store', default=1, type=int,
    help='number of edge features (deafult: 1)', metavar='NUM')

parser.add_argument(
    '--num-output-features', action='store', default=16, type=int,
    help='number of node features for NNConv (deafult: 16)', metavar='NUM')

parser.add_argument(
    '--job-name', action='store', default="NN_Conv", type=str,
    help="Job name for scheduler", metavar='NAME')

args = parser.parse_args()

kwargs = lbann.contrib.args.get_scheduler_kwargs(args)

MINI_BATCH_SIZE = args.mini_batch_size
NUM_EPOCHS = args.num_epochs
JOB_NAME = args.job_name
NUM_NODES = args.num_nodes
NUM_EDGES = args.num_edges
NUM_NODES_FEATURES = args.num_nodes_features
NUM_EDGE_FEATURES = args.num_edge_features
NUM_OUT_FEATURES = args.num_out_features


def graph_data_splitter(_input):

    split_indices = []

    start_index = 0
    split_indices.append(start_index)

    node_feature = NUM_NODES * NUM_NODES_FEATURES
    split_indices.append(node_feature)

    neighbor_features = NUM_EDGES * NUM_NODES_FEATURES
    split_indices.append(neighbor_features)

    edge_indices = NUM_EDGES
    split_indices.append(edge_indices)

    edge_features = NUM_EDGES * NUM_EDGE_FEATURES
    split_indices.append(edge_features)

    target = 1
    split_indices.append(target)

    split_indices = \
        [split_indices[i]+split_indices[i-1]
            for i in range(len(split_indices))]

    graph_input = lbann.slice(_input, axis=0,
                              slice_points=str_list(split_indices))
    graph_data_id = [lbann.Identity(graph_input) for x in range(5)]

    node_feature_dims = str_list([NUM_NODES, NUM_NODES_FEATURES])
    neighbor_feature_dims = str_list([NUM_EDGES, 1, NUM_NODES_FEATURES])
    edge_feature_dims = str_list([NUM_EDGES, NUM_EDGE_FEATURES])
    edge_indices_dims = str_list([NUM_EDGES, 1])
    target_dims = str_list([1])

    node_feature_mat = lbann.Reshape(graph_data_id[0],
                                     dims=node_feature_dims)
    neighbor_feature_mat = lbann.Reshape(graph_data_id[1],
                                         dims=neighbor_feature_dims)
    edge_feature_mat = lbann.Reshape(graph_data_id[2],
                                     dims=edge_feature_dims)
    edge_indices = lbann.Reshape(graph_data_id[3],
                                 dims=edge_indices_dims)
    target = lbann.Reshape(graph_data_id[4],
                           dims=target_dims)
    return \
        node_feature_mat, neighbor_feature_mat, edge_feature_mat, edge_indices, target


def reduction(graph_feature_matrix, channels):
    vector_size = str_list([1, NUM_NODES], name='Sum_Vector')
    reduction_vector = lbann.Constant(value=1,
                                      num_neurons=vector_size)
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
        [FC(128, name="NN_SQ_1"),
         lbann.Relu,
         FC(64, name="NN_SQ_2"),
         lbann.Relu,
         FC(32, name="NN_SQ_3"),
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
    node_fts = NNConvLayer(node_feature_mat,
                           neighbor_feature_mat,
                           edge_feature_mat,
                           edge_indices,
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
data_reader = data.Synthetic.make_data_reader("Synthetic_Scatter_Edge")
trainer = lbann.Trainer(mini_batch_size=MINI_BATCH_SIZE)

lbann.contrib.launcher.run(trainer,
                           model,
                           data_reader,
                           optimizer,
                           job_name=JOB_NAME,
                           **kwargs)
