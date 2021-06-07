import lbann
import os.path as osp

data_dir = osp.dirname(osp.realpath(__file__))


def make_data_reader(classname,
                     sample='get_sample_func',
                     num_samples='num_samples_func',
                     sample_dims='sample_dims_func'):
    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = 'python'
    _reader.role = 'train'
    _reader.shuffle = True
    _reader.percent_of_data_to_use = 1.0
    _reader.python.module = classname
    _reader.python.module_dir = data_dir
    _reader.python.sample_function = sample
    _reader.python.num_samples_function = num_samples
    _reader.python.sample_dims_function = sample_dims
    return reader


def make_graph_reader(filename,
                      num_samples,
                      max_num_nodes,
                      max_num_edges,
                      num_node_features,
                      num_edge_features,
                      has_edge_features=True):
    print(num_samples)
    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = 'graph'
    _reader.role = 'train'
    _reader.shuffle = True
    _reader.num_samples = num_samples
    _reader.percent_of_data_to_use = 1.0
    _reader.data_filedir = data_dir
    _reader.data_filename = filename
    _reader.graph.max_node_size = max_num_nodes
    _reader.graph.max_edge_size = max_num_edges
    _reader.graph.num_node_features = num_node_features
    _reader.graph.num_edge_features = num_edge_features
    _reader.graph.has_edge_features = has_edge_features
    return reader
