import numpy as np 


class Synthetic_Data:
    def __init__(self, num_samples = 1000):
        self.num_samples = num_samples
        
        self.data = np.random(num_samples, 19, 48,48,48)

    def __len__(self):
        return self.num_samples 
    
    def __getitem__(self, idx):
        return np.flatten(np.float32(self.data[idx]))        
