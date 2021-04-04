import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import NNConv


class LSC_Trainer(nn.Module):
    def __init__(self):
        super(LSC_Trainer, self).__init__()
        self.bond_encoder = BondEncoder(16)
        self.atom_encoder = AtomEncoder(64)
        self._graph_nn= nn.Sequential(nn.Linear(16, 1024, bias=False),
                                      nn.ReLU(),
                                      nn.Linear(1024, 256, bias=False),
                                      nn.ReLU(),
                                      nn.Linear(256, 64*32, bias=False))

        self.graph_conv = NNConv(64, 32, self._graph_nn)

        self._nn = nn.Sequential(nn.Linear(51*32, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 8),
                                 nn.ReLU(),
                                 nn.Linear(8, 1))

    def flatten(self, x):
        return x.view(x.size(0), -1)

    def forward(self, data):
        node_features = data.x
        edge_features = data.edge_attr
        edge_index = data.edge_index
        batch = data.batch

        encoded_atoms = self.atom_encoder(node_features)
        encoded_bonds = self.bond_encoder(edge_features)

        updated_features = self.graph_conv(encoded_atoms, edge_index, encoded_bonds)

        updated_features = self.flatten(to_dense_batch(updated_features, batch, max_num_nodes=51)[0])

        out = self._nn(updated_features)

        return out