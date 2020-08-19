import lbann 
import lbann.modules as nn
import os 
import os.path as osp 
import sys 

def make_data_reader():
    cur_dir = osp.dirname(osp.realpath(__file__))
    dataset_dir = osp.dirname(cur_dir)

   
    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = 'python'
    _reader.role = 'train'
    _reader.shuffle = True
    _reader.percent_of_data_to_use = 1.0
    _reader.python.module = 'dataset'
    _reader.python.module_dir = cur_dir
    _reader.python.sample_function = 'get_train'
    _reader.python.num_samples_function = 'num_train_samples'
    _reader.python.sample_dims_function = 'sample_dims'

def CNN3D_Model():
    ''' 3D CNN model from Fusion models for 
        Atomic and Molecular Structures
    '''
    input_ = lbann.Input(target_model = 'regression')
    
    data = lbann.Identity(input_)
    
    num_elements = (48 * 48 * 48 * 19)
    
    slice_points = str_list([0, num_elements, num_elements + 1])
    sliced_data = lbann.Slice(data, slice_points = slice_points)
    x = lbann.Identity(sliced_data, name = "sata_sample")
    y = lbann.Identity(sliced_data, name = "target")
    x = lbann.Reshapce(x, dims = "19 48 48 48")
    
    conv1 = nn.Convolution3DModule(out_channels = 64,
                                   kernel_size = 7)
    
    conv2 = nn.Convolution3DModule(out_channel = 64,
                                   kernel_size = 7)
    
    conv3 = nn.Convolution3DModule(out_channel = 64,
                                   kernel_size = 7)

    conv4 = nn.Convolution3DModule(out_channel = 128,
                                   kernel_size = 7)
    
    conv5 = nn.Convolution3DModule(out_channel = 256,
                                   kernel_size = 5)
    
    fc1 = nn.FullyConnectedModule(size = 10)
    fc2 = nn.FullyConnectedModule(size = 1)
 
    x = conv1(x)
    x = lbann.BatchNormalization(x)
    x_1 = lbann.Relu(x)

    x = conv2(x)
    x = lbann.Relu(x)
    x = lbann.BatchNormalization(x)
    x_2 = lbann.Sum(x, x_1)
    
    x = conv3(x)
    x = lbann.Relu(x)
    x = lbann.BatchNormalization(x)
    
    x_3 = lbann.Sum(x, x_1)
    
    x = conv4(x)
    x = lbann.Relu(x)
    x = lbann.BatchNormalization(x)
    
    x = conv5(x)
    x = lbann.Relu(x)
    x = lbann.BatchNormalization(x)
    
    x = lbann.Flatten(x)
    x = fc1(x)
    x = fc2(x)
    
    loss = lbann.ManAbsoluteError([x, y], name='MAE_loss')
    
    metrics = [lbann.Metric(loss, name = 'MAE') ]
    
def main():
    model = CNN3D_Model()
    opt = lbann.Adam(learn_rare = 1e-2,
                     beta1 = 0.9,
                     beta2 = 0.99, 
                     eps = 1e-8
                    )
    data_reader = make_data_reader() 
    trainer = lbann.Trainer(mini_batch_size = mini_batch_size,
                            name = "FAST_3DCNN")


if __name__ == '__main__':
    main()    



 

  
