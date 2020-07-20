import lbann 
import lbann.contrib.launcher 
import lbann.contrib.args

import argparse 
import os 

#import sparse_train
#import train
import GIN_train
import GCN_train
import Graph_train

import data.MNIST_Superpixel
import data.PROTEINS

desc = (" Training a Graph Convolutional Model using LBANN" )

parser = argparse.ArgumentParser(description=desc)

lbann.contrib.args.add_scheduler_arguments(parser) 
lbann.contrib.args.add_optimizer_arguments(parser) 

parser.add_argument(
    '--num-epochs', action='store', default=100, type=int,
    help='number of epochs (deafult: 100)', metavar='NUM')

parser.add_argument(
    '--mini-batch-size', action='store',default=32, type=int,
    help="mini-batch size (default: 32)", metavar='NUM')

parser.add_argument(
    '--dataset', action='store', default='MNIST', type=str,
    help="Dataset for model (default: MNIST)", metavar='NAME')

parser.add_argument(
    '--job-name', action='store', default="GCN_TEST", type=str,
    help="Job name for scheduler", metavar='NAME')

parser.add_argument(
    '--model', action = 'store', default='GCN', type=str,
    help="The type of model to use", metavar='NAME')

args = parser.parse_args()


kwargs = lbann.contrib.args.get_scheduler_kwargs(args) 

dataset = args.dataset
num_epochs = args.num_epochs
mini_batch_size = args.mini_batch_size 
job_name = args.job_name
model_arch = args.model


## Get Model

if (model_arch == 'GRAPH'):
    model = Graph_train.make_model(dataset = 'PROTEINS',
                            num_epochs = num_epochs)
elif(model_arch=='GIN'):
    model = GIN_train.make_model(dataset = 'PROTEINS',
                            num_epochs = num_epochs)
else:
    model = GCN_train.make_model(dataset = 'PROTEINS',
                            num_epochs=num_epochs)


optimizer = lbann.SGD(learn_rate = 1e-3)

#add logic for choosing a dataset 

data_reader = data.PROTEINS.make_data_reader()

trainer = lbann.Trainer(mini_batch_size = mini_batch_size)


lbann.contrib.launcher.run(trainer, model, data_reader, optimizer,
                           job_name = job_name,
                           **kwargs)