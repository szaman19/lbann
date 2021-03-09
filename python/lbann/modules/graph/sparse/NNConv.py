import lbann
from lbann.modules import Module, ChannelwiseFullyConnectedModule
from lbann.util import str_list


class NNConv(Module):
    """Details of the kernel is available at:
       "Neural Message Passing for Quantum Chemistry"
       https://arxiv.org/abs/1704.01212
    """
    global_count = 0

    def __init__(self,
                 sequential_nn,
                 num_nodes,
                 output_channels,
                 activation=lbann.Relu,
                 name=None):
        """Inititalize  the edge conditioned graph kernel with edge data
           represented with pseudo-COO format. The reduction over edge
           features are performed via the scatter layer

           The update function of the kernel is:

           ..  math::
                X^{\prime}_{i} = \Theta x_i + \sum_{j \in \mathcal{N(i)}}x_j \cdot h_{\Theta}(e_{i,j})

           where :math:`h_{\mathbf{\Theta}}` denotes a channel-wise NN module

        Args:
            sequential_nn ([Module] or (Module)): A list or tuple of layer
                                                  modules for updating the
                                                  edge feature matrix
            num_nodes (int): Number of vertices of each graph
                            (max number in the batch)
            output_channels (int): The output size of each node feature after
                                transformed with learnable weights
            activation (type): The activation function of the node features
            name (str): Default name of the layer is NN_{number}
        """
        NNConv.global_count += 1

        self.name = (name
                     if name
                     else 'NNConv_{}'.format(NNConv.global_count))

        self.output_channels = output_channels

        self.num_nodes = num_nodes

        self.node_activation = activation

        self.node_nn = \
            ChannelwiseFullyConnectedModule(self.output_channels,
                                            bias=False,
                                            activation=self.node_activation,
                                            name=self.name+"_node_weights")
        self.edge_nn = sequential_nn

    def forward(self, node_features, edge_features, edge_index):
        """Apply NNConv layer.

        Args:
            node_features (Layer); A 2D layer of node features of
                                   shape (num_nodes, input_channels)
            edge_features (Layer): A 2D layer of node features of
                                   shape (num_edges, edge_features)
            edge_index (Layer): A 1D layer of node features of
                                shape (num_edges * output_channels).
                                The indices used for reduction

        Returns:
            (Layer): The output after NNConv. The output layer has the shape
                     (num_nodes, self.output_channels)
        """

        node_feature_size = self.num_nodes * self.output_channels

        updated_node_features = self.node_nn(node_features)

        edge_update = self.edge_nn(edge_features)

        edge_values = \
            lbann.Reshape(edge_update,
                          dims=str_list([node_feature_size]),
                          name=self.name+"_edge_mat_reshape")

        edge_reduce = lbann.Scatter(edge_values,
                                    edge_index,
                                    dims=node_feature_size,
                                    name=self.name+"_aggregate")

        updated_edge_features = \
            lbann.Reshape(edge_reduce,
                          dims=str_list([self.num_nodes, self.output_channels]),
                          name=self.name+"_updated_edge_mat_reshape")

        update = lbann.Sum(updated_node_features,
                           updated_edge_features,
                           name=self.name+"_updated_node_features")

        return update
