from  data.synth_data import Synthetic_Data


training_data = Synthetic_Data(1000)

def get_train(index):
    return training_data[index]

def num_train_samples():
    return len(training_data)

def sample_dims():
    return (19*48*48*48, )

 
