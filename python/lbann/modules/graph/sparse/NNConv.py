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
                 num_edges,
                 input_channels,
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
                            (max number in the batch padded by 0)
            num_edges (int): Number of edges of each graph
                            (max in the batch padded by 0)
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
        self.input_channels = input_channels

        self.num_nodes = num_nodes
        self.num_edges = num_edges

        self.node_activation = activation

        self.node_nn = \
            ChannelwiseFullyConnectedModule(self.output_channels,
                                            bias=False,
                                            activation=self.node_activation,
                                            name=self.name+"_node_weights")
        self.edge_nn = sequential_nn

    def message(self,
                node_features,
                neighbor_features,
                edge_features):
        """Update node features and edge features. The Message stage of the
           convolution.

        Args:
            node_features (Layer); A 2D layer of node features of
                                   shape (num_nodes, input_channels)
            neighbor_features (Layer): A 3D layer of node features of
                                       shape (num_edges, 1, input_channels)
            edge_features (Layer): A 2D layer of edge features of
                                   shape (num_edges, edge_features)
        Returns:
            (Layer, Layer): Returns the updated node features and the messages
            for each node.
        """

        updated_node_features = self.node_nn(node_features)

        edge_update = None
        for layer in self.edge_nn:

            if edge_update:
                edge_update = layer(edge_update)
            else:
                edge_update = layer(edge_features)

        edge_values = \
            lbann.Reshape(edge_update,
                          dims=str_list([self.num_edges,
                                         self.input_channels,
                                         self.output_channels]),
                          name=self.name+"_edge_mat_reshape")
        edge_values = \
            lbann.MatMul(neighbor_features, edge_values)
        return updated_node_features, edge_values

    def aggregate(self,
                  edge_values,
                  edge_indices):
        """Aggregate the messages from the neighbors of the nodes

        Args:
            edge_values (Layer): A 2D layer of edge features of
                                   shape (num_edges, edge_features)
            edge_indices (Layer): A 1D layer of node features of
                                shape (num_edges * output_channels).
                                The indices used for reduction
        Returns:
            (Layer): A 2D layer of updated node features
        """

        node_feature_size = self.num_nodes * self.output_channels

        edge_values = lbann.Reshape(edge_values,
                                    dims=str_list([node_feature_size]))
        edge_reduce = lbann.Scatter(edge_values,
                                    edge_indices,
                                    dims=node_feature_size,
                                    name=self.name+"_aggregate")
        edge_reduce = lbann.Reshape(edge_reduce,
                                    dims=str_list([self.num_nodes,
                                                   self.output_channels]))
        return edge_reduce

    def forward(self,
                node_features,
                neighbor_features,
                edge_features,
                edge_index):
        """Apply NNConv layer.

        Args:
            node_features (Layer); A 2D layer of node features of
                                   shape (num_nodes, input_channels)
            neighbor_features (Layer): A 3D layer of node features of
                                       shape (num_edges, 1, input_channels)
            edge_features (Layer): A 2D layer of edge features of
                                   shape (num_edges, edge_features)
            edge_index (Layer): A 1D layer of node features of
                                shape (num_edges * output_channels).
                                The indices used for reduction

        Returns:
            (Layer): The output after NNConv. The output layer has the shape
                     (num_nodes, self.output_channels)
        """

        updated_node_fts, neighbor_vals = self.message(node_features,
                                                       neighbor_features,
                                                       edge_features)
        aggregated_fts = self.aggregate(edge_features, edge_index)

        update = lbann.Sum(updated_node_fts,
                           aggregated_fts,
                           name=self.name+"_updated_node_features")

        return update
