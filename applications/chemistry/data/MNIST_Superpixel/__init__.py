import torch 
import urllib.request
import tarfile 
import os
import os.path

import lbann 
 
data_dir = os.path.dirname(os.path.realpath(__file__))

def download_data():
    url = "http://ls7-www.cs.uni-dortmund.de/cvpr_geometric_dl/mnist_superpixels.tar.gz"
    training_name = "training.pt"
    test_name = "test.pt"

    files = [training_name, test_name]

    for f in files:
        data_file = os.path.join(data_dir, f)

        if not os.path.isfile(data_file): #File not in directory 
            tar_name = os.path.join(data_dir, "mnist_superpixel.tar.gz")

            if not os.path.isfile(tar_name):
                urllib.request.urlretrieve(url, filename=tar_name)
                extract_data()
            else:
                extract_data()
def extract_data():
     tar_name = os.path.join(data_dir, "mnist_superpixel.tar.gz") 
     print(tar_name)
     with tarfile.open(tar_name) as tar:
        tar.extractall()
        tar.close()

def make_data_reader(): #TO DO: Extend this to use this for validation / test set as well after testing 

    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = 'python'
    _reader.role = 'train'
    _reader.shuffle = False #Turn off shuffle for debugging 
    _reader.percent_of_data_to_use = 1.0 
    _reader.python.module = 'dataset'
    _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
    _reader.python.sample_function = 'get_train'
    _reader.python.num_samples_function = 'num_train_samples' 
    _reader.python.sample_dims_function = 'sample_dims' 

    return reader 

#if __name__ == '__main__':
    #download_data()
    #extract_data()
