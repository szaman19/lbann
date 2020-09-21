import os.path as osp 
import numpy as np 
import pandas as pd
import h5py

class PDB3DCNNDataset():
    '''
    Dataset for 3D voxel representation of processed  PDBBind data. 
    
    '''
    def __init__(self,
                 dataset = 'train',
                 cache_dir = '/usr/workspace/zaman2/pdbbind2016/',
                 checked_cached = True):
        self.dataset_type = None
        if (dataset == 'train' or dataset =='val' or dataset == 'test'):
            self.dataset_type = dataset
        else:
            raise ValueError('Incorrect dataset type. Use either train, val or test')
        
        self.fname_root = '/usr/workspace/zaman2/pdbbind2016/pdbbind2016_refined_pybel_processed'+\
        '_crystal_48_radius1_sigma1_rot0_'

        self.fname = self.fname_root + self.dataset_type + '.hdf'

        self.label_file = self.fname_root + 'info.csv'
        
        self.data = None 

        save_file = self.fname_root + self.dataset_type + '.npy'
        if (osp.isfile(save_file)):
            self.data = np.load(save_file)
        else:
            print("Saved numpy file not found. Generating dataset from hdf file. ")
            self.__generate_dataset()
        
    def __generate_dataset(self):
        h5file = h5py.File(self.fname, 'r')
        df  = pd.read_csv(self.label_file)
        
        self.data = []

        for i,ligand in enumerate(list(h5file.keys())):
            data_sample = h5file.get(ligand)[()]
            data_sample = np.transpose(data_sample, (3,0,1,2))
            data_sample = np.float32((data_sample.flatten()))
            
            label = df.loc[df['ligand_id']==ligand]['label'].item()
            label = np.float32(label)
            
            data_point = np.append(data_sample, label)

            self.data.append(data_point)
            
            print("Processing {} / {} ".format(i, len(list(h5file.keys()))),end="\r", flush=True)
            
            if (i == 100):
                break 

        self.data = np.array(self.data)
        
        fname = self.fname_root + self.dataset_type + '.npy'
        with open(fname, 'wb') as f:
            np.save(f, self.data)

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self,idx):
        
       return self.data[idx]

if __name__ == '__main__':
    dataset = PDB3DCNNDataset('train') 
    
    assert len(dataset[0]) == (48*48*48*19) + 1 


