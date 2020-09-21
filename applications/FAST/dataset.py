from  data.PDB_3DCNN_Dataset import PDB3DCNNDataset


training_data = PDB3DCNNDataset('train')

def get_train(index):
    return training_data[index]

def num_train_samples():
    return len(training_data)

def sample_dims():
    return ((19*48*48*48)+1, )

 
