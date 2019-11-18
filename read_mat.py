import h5py
import torch
import numpy as np
with h5py.File('../boxqp-master/H.mat', 'r') as f:
    print(list(f.keys()))

with h5py.File('../mycode/H.mat', 'r') as f:
    Q=np.array(list(f['HH']))
    p=np.array(list(f['f']))
    G=np.array(list(f['D']))
print(Q)
print(p)
