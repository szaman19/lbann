import h5py 
import numpy as np 

file_name ='pdbbind2016_refined_pybel_processed_crystal_48_radius1_sigma1_rot0_val.hdf'

f = h5py.File(file_name, 'r')
obj_name = list(f.keys())[0]
obj = f[obj_name]

print(type(obj))

print(obj.shape)
