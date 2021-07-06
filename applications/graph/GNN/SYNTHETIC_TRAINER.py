import lbann
import lbann.contrib.launcher
import lbann.contrib.args

import argparse
import os
import configparser
import data.Synthetic


from NNConvModel import make_model
desc = ("Training Edge-conditioned Graph Convolutional Model Using LBANN ")

parser = argparse.ArgumentParser(description=desc)

lbann.contrib.args.add_scheduler_arguments(parser)
lbann.contrib.args.add_optimizer_arguments(parser)

parser.add_argument(
    '--num-epochs', action='store', default=3, type=int,
    help='number of epochs (deafult: 3)', metavar='NUM')

parser.add_argument(
    '--mini-batch-size', action='store', default=2048, type=int,
    help="mini-batch size (default: 2048)", metavar='NUM')

parser.add_argument(
    '--num-edges', action='store', default=98, type=int,
    help='number of edges (deafult: 118)', metavar='NUM')

parser.add_argument(
    '--num-nodes', action='store', default=50, type=int,
    help='number of nodes (deafult: 51)', metavar='NUM')

parser.add_argument(
    '--num-node-features', action='store', default=9, type=int,
    help='number of node features (deafult: 9)', metavar='NUM')

parser.add_argument(
    '--num-edge-features', action='store', default=3, type=int,
    help='number of edge features (deafult: 3)', metavar='NUM')

parser.add_argument(
    '--num-out-features', action='store', default=32, type=int,
    help='number of node features for NNConv (deafult: 32)', metavar='NUM')

parser.add_argument(
    '--num-samples', action='store', default=10000, type=int,
    help='number of Samples (deafult: 10000)', metavar='NUM')


parser.add_argument(
    '--node-embeddings', action='store', default=100, type=int,
    help='dimensionality of node feature embedding (deafult: 100)', metavar='NUM')


parser.add_argument(
    '--edge-embeddings', action='store', default=16, type=int,
    help='dimensionality of edge feature embedding (deafult: 16)', metavar='NUM')

parser.add_argument(
    '--job-name', action='store', default="NN_Conv", type=str,
    help="Job name for scheduler", metavar='NAME')

args = parser.parse_args()

kwargs = lbann.contrib.args.get_scheduler_kwargs(args)

MINI_BATCH_SIZE = args.mini_batch_size
NUM_EPOCHS = args.num_epochs
JOB_NAME = args.job_name
NUM_NODES = 50
NUM_EDGES = 100
NUM_NODES_FEATURES = 9
NUM_EDGE_FEATURES = 3
NUM_OUT_FEATURES = args.num_out_features
NUM_SAMPLES = args.num_samples
EMBEDDING_DIM = args.node_embeddings
EDGE_EMBEDDING_DIM = args.edge_embeddings

# ----------------------------------------

# Generating configuration for dataset

# ----------------------------------------

config = configparser.ConfigParser()
config['Graph'] = {}
config['Graph']['num_nodes'] = str(NUM_NODES)
config['Graph']['num_edges'] = str(NUM_EDGES)
config['Graph']['num_node_features'] = str(NUM_NODES_FEATURES)
config['Graph']['num_edge_features'] = str(NUM_EDGE_FEATURES)
config['Graph']['num_samples'] = str(NUM_SAMPLES)

current_file = os.path.realpath(__file__)
app_dir = os.path.dirname(current_file)
_file_name = os.path.join(app_dir, 'config.ini')

with open(_file_name, 'w') as configfile:
    config.write(configfile)

os.environ['SYNTH_TEST_CONFIG_FILE'] = _file_name


model = make_model(NUM_NODES,
                   NUM_EDGES,
                   NUM_NODES_FEATURES,
                   NUM_EDGE_FEATURES,
                   EMBEDDING_DIM,
                   EDGE_EMBEDDING_DIM,
                   NUM_OUT_FEATURES,
                   NUM_EPOCHS)

fname = '/g/g92/zaman2/lbann/applications/graph/GNN/data/Synthetic/test_file.bin'
optimizer = lbann.SGD(learn_rate=1e-4)
# data_reader = data.Synthetic.make_graph_reader(fname,
#                                                NUM_SAMPLES,
#                                                NUM_NODES,
#                                                NUM_EDGES,
#                                                NUM_NODES_FEATURES,
#                                                NUM_EDGE_FEATURES,
#                                                True)
data_reader = data.Synthetic.make_data_reader("Synthetic_Sparse_Edge")
trainer = lbann.Trainer(mini_batch_size=MINI_BATCH_SIZE)

lbann.contrib.launcher.run(trainer,
                           model,
                           data_reader,
                           optimizer,
                           job_name=JOB_NAME,
                           lbann_args=['--num_io_threads=4'],
                           **kwargs)
