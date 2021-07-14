import lbann
from lbann.modules import Module, NNConv
from lbann.util import str_list
import lbann.modules as nn
import math
import warnings


class Sequential(Module):
    """Sequential container for LBANN layers. Similar to:
       https://pytorch.org/docs/stable/generated/\
       torch.nn.Sequential.html#torch.nn.Sequential

       Only supports layers in lbann.Module. Need to think up a kwargs
       trick to make it usable for all layers.
    """
    def __init__(self, sequential_layer_list):
        super(Sequential, self).__init__()
        self.layers = sequential_layer_list

    def forward(self, x):
        temp = x
        for layer in self.layers:
            temp = layer(temp)
        return temp


class global_add_pool(Module):
    """Combines the node feature matrix into a single vector with 
        addition along the column axis
        
    """
    global_count = 0

    def __init__(self,
                 mask=None,
                 num_nodes=None,
                 name=None):
        """

            params:
                mask (Layer): (default: None)
                num_nodes (int): (default : None)
                name (string): (default: None)
        """
        super().__init__()
        global_add_pool.global_count += 1
        self.name = (name if name else
                     "global_add_pool")
        if mask is None:
            if num_nodes is None:
                ValueError("Either requires one of mask or num_nodes must be set")
            self.reduction = lbann.Constant(value=1,
                                            num_neurons=str_list([1, num_nodes]),
                                            name=self.name)
        else:
            if num_nodes is None:
                warnings.warn("Only one of mask or num_nodes should be set. Using mask value")
            self.reduction = mask 

    def forward(self, x):
        return lbann.MatMul(self.reduction, x, name=self.name + "_matmul")


class Graph_Conv(Module):
    global_count = 0
    """docstring for Graph_Conv"""
    def __init__(self,
                 input_feature_dim,
                 output_feature_dim,
                 num_layers,
                 num_nodes,
                 num_edges,
                 edge_nn=None,
                 name=None):
        super(Graph_Conv, self).__init__()
        Graph_Conv.global_count += 1
        self.output_channel_dim = output_feature_dim
        self.input_channel_dim = input_feature_dim
        self.num_layers = num_layers
        self.num_nodes= num_nodes
        self.num_edges = num_edges

        self.edge_conv = NN_Conv(edge_nn,
                                 num_nodes,
                                 num_edges,
                                 input_feature_dim,
                                 output_channels,
                                 )
        self.name = (name if name else
                     'GraphConv_{}'.format(Graph_Conv.global_count))

        self.rnn = nn.GRU(output_channels)

        self.weights = []

        for i in range(num_layers):

            weight_init = \
                lbann.Weights(initializer=lbann.UniformInitializer(min=-1/(math.sqrt(output_channels)), 
                                                                   max=1/(math.sqrt(output_channels))))

            self.weights.append(nn.ChannelwiseFullyConnectedModule(output_channels, 
                                                                   bias=False, 
                                                                   weights=weight_init,
                                                                   name=f"{self.name}_{i}_weight"))

    def forward(self,
                node_features,
                node_feature_target_indices,
                edge_features,
                edge_source_indices,
                edge_target_indices):
        
        if (input_feature_dim < self.output_channels):

            num_zeros = self.output_channels - input_feature_dim

            zeros = lbann.Constant(value=0,
                                   num_neurons=str_list([num_nodes, num_zeros]))

            node_features = lbann.Concatenation(node_features, zeros, axis=1)
      
        for layer in range(self.num_layers):
            nf_clone = lbann.Identity(lbann.Split(node_features))
            
            messages = self.weights[layer](nf_clone)

            aggregate = lbann.Scatter(adjacency_mat, messages, self.output_channels)

            Node_FT_GRU_input = aggregate


            # Update node_features according to edge convolutions

            # Generate the neighbor matrices with gather 

            neighbor_features = lbann.Reshape(lbann.Gather(node_features, edge_source_indices),
                                              dims=str_list([num_edges, 1, self.output_channels]))
            Node_FT_GRU_PrevState = self.edge_conv(node_features,
                                                   neighbor_features,
                                                   edge_features,
                                                   edge_target_indices)

            node_features = self.rnn(Node_FT_GRU_input, Node_FT_GRU_PrevState)
        
        return node_features


class Graph_Attention(Module):
    """docstring for Graph_Attention"""
    global_count = 0

    def __init__(self,
                 feat_size,
                 output_size,
                 name=None):
        super(Graph_Attention, self).__init__()
        Graph_Attention.global_count += 1
        self.nn_1 = Sequential([nn.FullyConnectedModule(feat_size),
                                lbann.Softsign,
                                nn.FullyConnectedModule(output_size),
                                lbann.Softsign
                                ])
        self.nn_2 = nn.FullyConnectedModule(output_size,
                                            activation=lbann.Softsign)
        self.name = (name if name else
                     'GraphAttention_{}'.format(Graph_Attention.global_count))

    def forward(self,
                updated_nodes,
                original_nodes,
                num_nodes):
        num_nodes = original_nodes.size(0)
        for i in range(num_nodes):
            concat = lbann.Concatenation(original_nodes[i],
                                         updated_nodes[i])
            attention_vector = self.nn_1(concat)
            attention_score = lbann.Softmax(attention_vector,
                                            name=self.name+"_softmax_{}".format(i))
            updated_nodes[i] = self.nn_2(updated_nodes[i])
            updated_nodes[i] = lbann.Multiply(attention_score,
                                              updated_nodes[i],
                                              name=self.name+"_output_{}".format(i))
            updated_nodes[i] = lbann.Reshape(updated_nodes[i], dims="{} {}".format(1,updated_nodes.size(1)))
        return updated_nodes


class SGCNN(Module):
    """docstring for SGCNN"""
    def __init__(self,
                 num_nodes=4,
                 input_channels=19,
                 out_channels=1,
                 covalent_out_channels=20,
                 covalent_layers=1,
                 noncovalent_out_channels=30,
                 noncovalent_layers=1):
        super(SGCNN, self).__init__()
        self.num_nodes = num_nodes
        self.input_channels = input_channels

        cov_out = covalent_out_channels
        noncov_out = noncovalent_out_channels
        covalent_edge_nn = \
            Sequential([nn.FullyConnectedModule(int(cov_out/2)),
                        lbann.Softsign,
                        nn.FullyConnectedModule(cov_out*cov_out),
                        lbann.Softsign
                        ])
        noncovalent_edge_nn = \
            Sequential([nn.FullyConnectedModule(int(noncov_out/2)),
                        lbann.Softsign,
                        nn.FullyConnectedModule(noncov_out*noncov_out),
                        lbann.Softsign
                        ])

        self.covalent_propagation = Graph_Conv(covalent_out_channels,
                                               covalent_layers,
                                               num_nodes,
                                               covalent_edge_nn)
        self.non_covalent_propagation = Graph_Conv(noncovalent_out_channels,
                                                   noncovalent_layers,
                                                   num_nodes,
                                                   noncovalent_edge_nn)

        self.add_pool_vector =  \
            lbann.Constant(value=1,
                           num_neurons=str_list([1, num_nodes]),
                           name="Reduction_Vector_SGCNN")
        self.cov_attention = Graph_Attention(covalent_out_channels,
                                             covalent_out_channels)
        self.noncov_attention = Graph_Attention(noncovalent_out_channels,
                                                noncovalent_out_channels)
        self.fully_connected_mlp = \
            Sequential([nn.FullyConnectedModule(int(noncov_out/1.5)),
                        lbann.Relu,
                        nn.FullyConnectedModule(int(noncov_out/2)),
                        lbann.Relu,
                        nn.FullyConnectedModule(out_channels)])
        self.gap = global_add_pool(num_nodes)

    def forward(self,
                x,
                covalent_adj,
                non_covalent_adj,
                edge_features,
                edge_adjacencies,
                ligand_id_matrix,
                fused=False):
        x_cov = self.covalent_propagation(x,
                                          covalent_adj,
                                          edge_features,
                                          edge_adjacencies)
        x = self.cov_attention(x_cov, x)
        x_noncov = self.non_covalent_propagation(x,
                                                 non_covalent_adj,
                                                 edge_features,
                                                 edge_adjacencies)
        x = self.noncov_attention(x_noncov, x)
        x = x.get_mat()
        ligand_only = lbann.MatMul(ligand_id_matrix, x)
        x = self.gap(ligand_only)
        if(fused):
            return x
        else:
            x = self.fully_connected_mlp(x)
            return x
