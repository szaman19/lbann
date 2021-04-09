import numpy as np
import glob 
import pickle

_files = glob.glob("/p/vast1/zaman2/synth_data/*.pickle")

#print(_files)


_vals = {}

for _file in _files:
    _temp = _file.split("/")[-1].split(".")[0].split("_")
    _num_nodes = _temp[0]
    _num_edges = _temp[1]
    print(_num_nodes, _num_edges)
    
    _edge_dic = {}
    with open(_file, 'rb') as f:
        _data = pickle.load(f)
        
        edge_sizes = []
        for obj in _data:
            edge_sizes.append(obj.num_edges)
        _edge_dic[_num_edges] = edge_sizes
    _vals[_num_nodes] = _edge_dic

with open("_gen_stats.pickle", 'wb') as f:
    pickle.dump(_vals, f)
