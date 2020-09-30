import argparse
import lbann 
import lbann.modules as nn
import os 
import os.path as osp 
import sys 
import lbann.contrib.launcher 
import lbann.contrib.args 

from lbann.util import str_list 

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
    _reader.python.module_dir = dataset_dir
    _reader.python.sample_function = 'get_train'
    _reader.python.num_samples_function = 'num_train_samples'
    _reader.python.sample_dims_function = 'sample_dims'

    return reader

def CNN3D_Model( num_epochs = 100, 
                 callbacks = []):
    ''' 3D CNN model from Fusion models for 
        Atomic and Molecular Structures
    '''
    input_ = lbann.Input(target_mode = 'N/A')
    
    data = lbann.Identity(input_)
    
    num_elements = (48 * 48 * 48 * 19)
    
    slice_points = str_list([0, num_elements, num_elements + 1])
    sliced_data = lbann.Slice(data, slice_points = slice_points)
    x = lbann.Identity(sliced_data, name = "sata_sample")
    y = lbann.Identity(sliced_data, name = "target")
    x = lbann.Reshape(x, dims = "19 48 48 48")
       
    fc1 = nn.FullyConnectedModule(size = 10)
    fc2 = nn.FullyConnectedModule(size = 1)
 
    conv_1_kernel = str_list([7,7,7])
    conv_1_res_1_kernel = str_list([7,7,7])
    conv_1_res_2_kernel = str_list([7,7,7])
    conv_2_kernel = str_list([7,7,7])
    conv_3_kernel = str_list([5,5,5])
    
   
    conv_1_stride = str_list([2,2,2])
    conv_1_res_1_stride = str_list([1,1,1])
    conv_1_res_2_stride = str_list([1,1,1])
    conv_2_stride = str_list([3,3,3])
    conv_3_stride = str_list([2,2,2])

    avg_pool3d_ksize = str_list([2,2,2])
    avg_pool3d_stride = str_list([2,2,2])

    zero_padding = str_list([0,0,0])
    x = lbann.Convolution(x,
                          num_dims = 3,
                          num_output_channels = 64,
                          num_groups = 1, 
                          conv_dims = conv_1_kernel,
                          conv_strides = conv_1_stride,
                          conv_pads = str_list([3,3,3]),
                          has_bias = True,
                          has_vectors = True,
                          name="Conv_1")
   
    x = lbann.Relu(x,
                     name = "Relu_1")
    
    x_1 = lbann.BatchNormalization(x,
                                 name = "BN_1")
    x = lbann.Convolution(x_1,
                          num_dims = 3, 
                          num_output_channels = 64, 
                          num_groups = 1, 
                          conv_dims = conv_1_res_1_kernel, 
                          conv_strides = conv_1_res_1_stride,
                          conv_pads = str_list([3,3,3]),
                          has_bias = True,
                          has_vectors = True,
                          name="Conv_1_res_1")
    x = lbann.Relu(x,
                   name="Relu_res_1")
    x = lbann.BatchNormalization(x,
                                 name="BN_Res_1")
    
    x_2 = lbann.Sum(x, x_1, name="Conv_Layer_1_+Conv_Layer_Res_1")
    
    x = lbann.Convolution(x_2,
                          num_dims = 3,
                          num_output_channels = 64,
                          num_groups = 1, 
                          conv_dims = conv_1_res_2_kernel,
                          conv_strides = conv_1_res_2_stride, 
                          conv_pads = str_list([3,3,3]),
                          has_bias = True,
                          has_vectors = True,
                          name="Conv_1_res_2")
    x = lbann.Relu(x,
                   name="Relu_res_2")
    
    x = lbann.BatchNormalization(x,
                                 name="BN_res_2")
    
    x_3 = lbann.Sum(x, x_1, name="Conv_Layer_1+Conv_Layer_3")
    
    x = lbann.Convolution(x_3,
                          num_dims = 3,
                          num_output_channels = 96,
                          num_groups = 1,
                          conv_dims = conv_2_kernel, 
                          conv_strides = conv_2_stride,
                          conv_pads = zero_padding,
                          has_bias = True,
                          has_vectors = True, 
                          name = "Conv_2")
    x = lbann.Relu(x,
                   name="Relu_2")
    x = lbann.BatchNormalization(x,
                                 name="BN_2")
    
    x = lbann.Pooling(x,
                      num_dims = 3,
                      pool_dims = avg_pool3d_ksize,
                      pool_strides = avg_pool3d_stride,
                      pool_pads = zero_padding,
                      has_vectors = True,
                      pool_mode = "average_no_pad",
                      name = "avg_pooling_1")

    x = lbann.Convolution(x,
                          num_dims = 3,
                          num_output_channels = 128, 
                          num_groups = 1, 
                          conv_dims = conv_3_kernel,
                          conv_strides = conv_3_stride, 
                          conv_pads = str_list([1,1,1]),
                          has_bias = True, 
                          has_vectors = True, 
                          name = "Conv_3")
    x = lbann.Relu(x,
                   name="Relu_3")
    x = lbann.BatchNormalization(x,
                                 name="BN_3")
    
    x = lbann.Pooling(x,
                      num_dims = 3,
                      pool_dims = avg_pool3d_ksize,
                      pool_strides = avg_pool3d_stride,
                      pool_pads = str_list([1,1,1]), 
                      has_vectors = True,
                      pool_mode = "average_no_pad",
                      name = "avg_pooling_2")

    x = fc1(x)
    x = fc2(x)
    
    loss = lbann.MeanAbsoluteError([x, y], name='MAE_loss')
    
    layers = lbann.traverse_layer_graph(input_)
    metrics = [lbann.Metric(loss, name = 'MAE') ]
    
    model = lbann.Model(num_epochs, 
                        layers,
                        objective_function = loss,
                        metrics = metrics, 
                        callbacks = callbacks)
    return model

desc = ("Training 3D-CNN on PDBBind Data using LBANN")

parser = argparse.ArgumentParser(description = desc)

parser.add_argument(
    '--job-name', action='store', default='mofae', type=str,
    help='job name', metavar='NAME')
parser.add_argument(
    '--mini-batch-size', action='store', default=128, type=int,
    help='mini-batch size (default: 128)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=100, type=int,
    help='number of epochs (default: 100)', metavar='NUM')

lbann.contrib.args.add_scheduler_arguments(parser)
args = parser.parse_args()

kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
                       
def main():
    
    num_epochs = args.num_epochs
    mini_batch_size = args.mini_batch_size

    opt = lbann.Adam(learn_rate = 1e-2,
                     beta1 = 0.9,
                     beta2 = 0.99, 
                     eps = 1e-8
                    )
    data_reader = make_data_reader() 
    
    trainer = lbann.Trainer(mini_batch_size = mini_batch_size,
                            name = "FAST_3DCNN")
    

    print_model = lbann.CallbackPrintModelDescription() #Prints initial Model after Setup
    training_output = lbann.CallbackPrint( interval = 1,
    print_global_stat_only = False) #Prints training progress
    gpu_usage = lbann.CallbackGPUMemoryUsage()
    timer = lbann.CallbackTimer()
    
    callbacks = [print_model, training_output, gpu_usage, timer]
    
    model = CNN3D_Model(num_epochs, callbacks)

    lbann.contrib.launcher.run(trainer, model, data_reader, opt, **kwargs)

if __name__ == '__main__':
    main()    



 

  
