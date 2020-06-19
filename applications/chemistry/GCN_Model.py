import lbann 
from lbann.util import str_list



#asdsad



def make_model(num_vertices = None, 
               node_features = None, 
               num_classes = None,
               dataset = None,
               num_epochs = 1):
    
    '''
    
    Construct a simple single layer GCN Model. 

    '''

    assert num_vertices != dataset #Ensure atleast one of the values is set 

    if dataset is not None:
        assert num_vertices is None

        if dataset == 'MNIST':
            num_vertices = 75
            num_classes = 10
            node_features = 1

        elif dataset == 'Synthetic':
            num_vertices = 3
            num_classes = 2
            node_features = 2
        else:
            raise Exception("Unkown Dataset")

    assert num_vertices is not None
    assert num_classes is not None 
    assert node_features is not None 
    

    #----------------------------------
    # Reshape and Slice Input Tensor 
    #----------------------------------

    input_ = lbann.Input(target_mode = 'classification')

    # Input dimensions should be (num_vertices * node_features + num_vertices^2 + num_classes )
    
    # input should have atleast two children since the target is classification 

    
    sample_dims = num_vertices*node_features + (num_vertices ** 2) + num_classes
    graph_dims = num_vertices*node_features + (num_vertices ** 2)
    
    graph = lbann.Identity(lbann.Slice(input_, axis = 0,slice_points = str_list([0, graph_dims])))
    target = lbann.Identity(lbann.Slice(input_, axis = 0,slice_points=str_list([graph_dims, sample_dims])))

    # Slice graph into node feature matrix, and adjacency matrix 

    feature_matrix_size = num_vertices * node_features 

    #slice

    feature_matrix = lbann.Identity(lbann.Slice(graph, axis=0, slice_points = str_list([0, feature_matrix_size]))) 
    adj_matrix = lbann.Identity(lbann.Slice(graph, axis=0, slice_points = str_list([feature_matrix_size, graph_dims])))
    
    #reshape 

    feature_matrix = lbann.Reshape(feature_matrix, dims = str_list([num_vertices, num_vertices]), name="Node_features")
    adj_matrix = lbann.Reshape(adj_matrix, dims = str_list([num_vertices,num_vertices]), name="Adj_Mat") 


    #----------------------------------
    # Perform Graph Convolution
    #----------------------------------

    # To Do: Implement lbann.GCN()
    # 
    # x = lbann.GCN(feature_matrix, adj_matrix, output_channels = N) # X is now the feature_matrix of shape num_vertices x output_channels 
    #
    
    print("Warning: Not using GCN layer and forwaring orignal feature matrix to reduction step. Should only use this when testing dataset / data reader")
    
    x = feature_matrix # Place holder while GCN is implemented
    out_channel = node_features
    
    #----------------------------------
    # Apply Reduction on Node Features
    #----------------------------------

    average_vector = lbann.Constant(value = 1/num_vertices, num_neurons = str(out_channel))
    x = lbann.MatMul(x, average_vector) # X is now a vector with output_channel dimensions 
    
    x = lbann.FullyConnected(x, num_neurons=num_classes, name="Output_FullyConnected")
    
    layers = lbann.traverse_layer_graph(input_)

    #----------------------------------
    # Loss Function and Accuracy s
    #----------------------------------
    
    
    probs = lbann.Softmax(x)
    loss = lbann.CrossEntropy(probs, target)
    accuracy = lbann.CategoricalAccuracy(probs, target)

    layers = lbann.traverse_layer_graph(input_)
    
    print_model = lbann.CallbackPrintModelDescription() #Prints initial Model after Setup

    training_output = lbann.CallbackPrint( interval = 1,
                           print_global_stat_only = False) #Prints training progress
    gpu_usage = lbann.CallbackGPUMemoryUsage()
    
    callbacks = [print_model, training_output, gpu_usage]

    metrics = [lbann.Metric(accuracy, name='accuracy', unit="%")]

    model = lbann.Model(num_epochs, 
                       layers = layers,
                       objective_function = loss,
                       metrics = metrics, 
                       callbacks = callbacks
                       )
    return model


if __name__ == '__main__':
    model = make_model()
    
