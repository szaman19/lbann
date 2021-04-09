from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.data import Data
from tqdm import tqdm
import torch
import pickle
import  multiprocessing as mp
import numpy as np

node_sizes  = [64, 128]
p_vals  = np.logspace(-2.5,-.4, num=10)

_f_name = "/p/vast1/zaman2/synth_data/{}_{}_Pytorch.pickle"

_num_samples = 10000

def make_dataset(_n, _p):

    _dataset = []
    max_eddges = 0
    edge_spread = []
    count = 0
    while(count < _num_samples):
            
        edge_indices = erdos_renyi_graph(_n, _p)
        
        if (edge_indices.shape[1] > 1):
            node_features = torch.randint(1,(_n, 9), dtype=torch.int)
            edge_features = torch.randint(1,(edge_indices.shape[1], 3), dtype=torch.int)
            
            target = torch.rand(1,1)
            data = Data(x=node_features,
                            edge_index = edge_indices,
                             edge_attr=edge_features,
                             y = target)
            _dataset.append(data)
            edge_spread.append(edge_indices.shape)
            if edge_indices.shape[1] > max_eddges:
                max_eddges = edge_indices.shape[1]
            count += 1
    with open(_f_name.format(_n,max_eddges), 'wb') as f:
        pickle.dump(_dataset, f)
    print(max_eddges)
    
    edge_spread = np.array(edge_spread)

    np.save(f'{_n}_{max_eddges}.npy', edge_spread)
    return(max_eddges)
    


combos = []
for _n in node_sizes:
    for _p in p_vals:
    
        make_dataset(_n,_p)
        print(_n, _p)
