import torch 
import os 
import os.path
import lbann #may not need this 

import numpy as np 
from torch.utils.data import Dataset 



data_dir = os.path.dirname(os.path.realpath(__file__))

train_file_path = os.path.join(data_dir, 'training.pt')
test_file_path = os.path.join(data_dir, 'test.pt')


class MNIST_Superpixel_Dataset(Dataset):
    
    def __init__(self, train=True, processed=False):
        super(MNIST_Superpixel_Dataset, self).__init__()
        self.num_vertices = 75 #All graphs have 75 nodes 
        if (train):
            self.num_data = 60000
        else:
            self.num_data = 20000

        if (processed):
            self.node_features, self.positions, self.edges,  self.targets = self.load_processed_training()
        else:
            self.node_features, self.positions, self.edges, self.targets = self.process_training_data()
        #print(self.node_features.shape)
        #print(self.edges.shape)
        #print(self.positions.shape)
        #print(self.targets.shape)
    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        data_x = self.node_features[idx].flatten()
        data_edges = self.edges[idx].flatten()
        data_target = self.targets[idx].flatten()
        #print(data_x.shape)
        #print(data_edges.shape)
        #print(data_target.shape)
        return np.concatenate([data_x, data_edges, data_target])


    def process_training_data(self): # Process Training File
        node_features, edge_index, edge_slices, positions, y = torch.load(train_file_path)
    
        assert y.size(0) == node_features.size(0)
        assert y.size(0) == positions.size(0)
        assert y.size(0) == self.num_data ## 

        
        #node_features, positions = node_features.view(self.num_data * self.num_vertices, 1), \
        #                            positions.view(self.num_data * self.num_vertices, 2)


        # Nodes features should be (60000, 75)
        
        node_features = np.float32(node_features)
        
        # Position should be (60000, 75, 2)

        positions = np.float32(positions)

        # Convert edge_index to edge matrix representation with shape (60000, 75, 75)
        
        adj_matrices = np.zeros( (self.num_data, self.num_vertices, self.num_vertices), dtype=np.float)

        assert (self.num_data + 1) == edge_slices.size(0), "Expected: {}, Got{} ".format(self.num_data + 1, edge_slices.size(0))
        
        for slice_index in range(self.num_data):
            print("{}/{} completed \r".format(slice_index+1, self.num_data), end='',flush=True)
            start_index = edge_slices[slice_index]
            end_index = edge_slices[slice_index + 1]

            graph_num = slice_index
            elist = edge_index[:, start_index: end_index ]

            adj_matrices[graph_num] = self.edge_list_to_dense(elist)


        # Convert y to target with one hot encoding and shape (60000, 10)

        targets = np.zeros ( (self.num_data, 10), dtype=np.float)

        for i, target in enumerate(y):
            print("{}/{} completed".format(i+1, len(y)), end='') 
            targets[i][target] = 1

        np.save('node_features.npy',node_features)
        np.save('positions.npy',positions)
        np.save('adj_matrices.npy', adj_matrices)
        np.save('targets.npy',targets)
        
        
        return node_features, positions, adj_matrices, targets 
    def edge_list_to_dense(self, elist):
        adj_mat = np.zeros((self.num_vertices, self.num_vertices), dtype=np.float)

        ## elist should be of shape (2, num_edges) 

        num_edges = elist.size(1)

        for edge in range(num_edges):
            source, sink = elist[:,edge]
            source = source.item()
            sink = sink.item()
            adj_mat[source][sink] = 1.0
            adj_mat[sink][source] = 1.0
        return adj_mat
    def load_processed_training(self):
        node_features = np.load('node_features.npy')
        positions = np.load('positions.npy')
        adj_matrices = np.load('adj_matrices.npy')
        targets = np.load('targets.npy')

        return node_features, positions, adj_matrices, targets 


training_data = MNIST_Superpixel_Dataset(train=True, processed=True)

def get_train(index):
    return training_data[index]


def num_train_samples():
    return len(training_data, processed=True)


def sample_dims():
    adjacency_matrix_size = 75 * 75 
    node_feature_size = 75 
    target_size = 10
    return (adjacency_matrix_size + node_feature_size + target_size)


if __name__ == '__main__':
    dataset = MNIST_Superpixel_Dataset(processed=True)
