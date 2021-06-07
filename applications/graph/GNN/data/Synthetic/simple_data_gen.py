import struct
import random
import numpy as np 

num_graphs = 10000
node_features = 9
edge_features = 3 

node_sizes = []
edge_sizes = [] 

tg_i = []
sg_i = []

nf = []
ef = []

t = np.zeros((num_graphs,1))
with open("test_file.bin",'wb') as f:
    for i in range(num_graphs):
        num_nodes = random.randint(10,50)
        num_edges = num_nodes + random.randint(num_nodes // 2, num_nodes)

        f.write(struct.pack('i', num_nodes))
        f.write(struct.pack('i', num_edges))
    
        node_sizes.append(num_nodes)
        edge_sizes.append(num_edges)

        node_feats = [random.randint(0, 3) for x in range(num_nodes * node_features)]
        edge_feats = [random.randint(0, 3) for x in range(num_edges * edge_features)]

        nf.append(np.array(node_feats))
        ef.append(np.array(edge_feats))

        target_indices = [random.randint(0, num_nodes) for x in range(num_edges)]
        source_indices = [random.randint(0, num_nodes) for x in range(num_edges)]

        tg_i.append(np.array(target_indices))
        sg_i.append(np.array(source_indices))

        target = random.random()
        t[i] = target

        f.write(struct.pack('f'*len(node_feats), *node_feats))
        f.write(struct.pack('f'*len(edge_feats), *edge_feats))
        f.write(struct.pack('i'*num_edges, *source_indices))
        f.write(struct.pack('i'*num_edges, *target_indices))
        f.write(struct.pack('f', target))


with open('graph_attrs.txt', 'w') as f:
    for nn, ne in zip(node_sizes, edge_sizes):
        f.write(f'{nn},{ne}\n')

max_nodes = max(node_sizes)
max_edges = max(edge_sizes)
print(f"max nodes  {max_nodes}, max edges {max(edge_sizes)}")


nfts = np.zeros((num_graphs, max_nodes * node_features))
efts = np.zeros((num_graphs, max_edges * edge_features))

tis = -1 * np.ones((num_graphs, max_edges))
sis = -1 * np.ones((num_graphs, max_edges))

for i in range(num_graphs):
    nfts[i][0:len(nf[i])] = nf[i]
    efts[i][0:len(ef[i])] = ef[i]

    tis[i][0:len(tg_i[i])] = tg_i[i]
    sis[i][0:len(sg_i[i])] = sg_i[i]

data = np.hstack((nfts, efts, sis, tis, t))

np.save("python_npy_test.npy", data)
print(data.shape)