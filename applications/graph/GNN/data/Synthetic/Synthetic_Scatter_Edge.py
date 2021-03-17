import numpy as np
import pickle
import os.path as osp


class Synthetic_Scatter_Edge(object):
    """docstring for Synthetic_Scatter_Edge"""
    def __init__(self,
                 num_samples,
                 num_nodes,
                 node_features,
                 edge_features,
                 max_edges=None,
                 use_cached=True,
                 cache_data=True,
                 cached_file=None):
        super(Synthetic_Scatter_Edge, self).__init__()
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.edge_features = edge_features
        self.max_edges = max_edges
        self.cache_data = cache_data
        self.dataset = None

        if (use_cached):
            if (cached_file):
                self.dataset = np.load(cached_file)
                print("Using cached data")
            else:

                _file_string = "synth_scatter_graphs_{}_{}_{}_{}_{}.p".format(num_samples,
                                                                              num_nodes,
                                                                              max_edges,
                                                                              node_features,
                                                                              edge_features)
                data_dir = osp.dirname(osp.realpath(__file__))
                _file_string = osp.join(data_dir, _file_string)
                try:
                    with open(_file_string, 'rb') as f:
                        self.dataset = pickle.load(f)
                except IOError:
                    print("File not found. Generating dataset")
                    self.generate_data()
        else:
            self.generate_data()

    def generate_data(self):
        """Generate
        """
        node_features = np.random.random((self.num_samples,
                                          self.num_nodes,
                                          self.node_features))

        edge_indices, neighbor_fts, edge_fts = \
            self.generate_edges(node_features)

        targets = np.random.random((self.num_samples, 1))
        self.dataset = [(node_features[i], neighbor_fts[i], edge_indices[i], edge_fts[i], targets[i]) for
                        i in range(self.num_samples)]
        _file_string = \
            "synth_scatter_graphs_{}_{}_{}_{}_{}.p".format(self.num_samples,
                                                           self.num_nodes,
                                                           self.max_edges,
                                                           self.node_features,
                                                           self.edge_features)
        with open(_file_string, 'wb') as f:
            pickle.dump(self.dataset, f)

    def generate_edges(self, node_fts):

        max_edges = 0
        edge_indices = []

        neighbor_fts = []
        edge_fts = []

        for i in range(self.num_samples):
            node_index = []

            graph_edges = 0

            neighbor_ft = []
            for n in range(self.num_nodes):
                nodes_choices = list(range(self.num_nodes))
                nodes_choices.remove(n)  # no self loops

                num_neighbors = np.random.choice(self.num_nodes - 1)

                node_adj = np.unique(np.random.choice(nodes_choices,
                                                      num_neighbors))
                num_neighbors = len(node_adj)

                node_indices = n * np.ones((2, num_neighbors))
                neighbor_ft.append(node_fts[i][node_adj])
                node_indices[1, :] = node_adj
                node_index.append(node_indices)

                graph_edges += num_neighbors

            if (max_edges < graph_edges):
                max_edges = graph_edges

            edge_feature = np.random.random((graph_edges, self.edge_features))

            edge_indices.append(np.concatenate(node_index, axis=1))
            neighbor_ft = np.concatenate(neighbor_ft, axis=0)
            neighbor_fts.append(neighbor_ft)

            edge_fts.append(edge_feature)

        padded_neighbor_fts = np.zeros((self.num_samples,
                                        max_edges,
                                        self.node_features))
        padded_edge_fts = np.zeros((self.num_samples,
                                    max_edges,
                                    self.edge_features))

        padded_edge_indices = -1 * np.ones((self.num_samples, 2, max_edges))

        for i, edge_index in enumerate(edge_indices):
            num_edges = edge_index.shape[1]
            padded_edge_indices[i, :2, :num_edges] = edge_index
            padded_neighbor_fts[i, : num_edges, :] = neighbor_fts[i]
            padded_edge_fts[i, : num_edges, :] = edge_fts[i]
        self.max_edges = max_edges

        return padded_edge_indices, padded_neighbor_fts, padded_edge_fts

    def get_sample(self, i):
        return self.dataset[i]


number_samples = 10000
number_nodes = 10
number_node_features = 10
number_edge_features = 1
max_edges = 53

dataset = Synthetic_Scatter_Edge(number_samples,
                                 number_nodes,
                                 number_node_features,
                                 number_edge_features,
                                 max_edges=max_edges)


def get_sample_func(index):
    node_fts, neighbor_fts, edge_ind, edge_fts, target = dataset.get_sample(index)

    _data = np.concatenate([node_fts.flatten(),
                            neighbor_fts.flatten(),
                            edge_ind[0].flatten(),
                            edge_fts.flatten(),
                            target])
    _data = np.float32(_data)
    return _data


def num_samples_func():
    return number_samples


def sample_dims_func():
    node_feature_size = number_nodes * number_node_features
    neighbor_features_size = dataset.max_edges * number_node_features
    edge_indices_size = dataset.max_edges
    edge_features_size = dataset.max_edges * number_edge_features
    return (node_feature_size + neighbor_features_size + edge_indices_size + edge_features_size + 1,)


if __name__ == '__main__':
  print(num_samples_func())
  print(sample_dims_func())
  print(get_sample_func(0).shape)
  print(dataset.max_edges)

  for i in range(num_samples_func()):
    print(get_sample_func(i).shape)
    
