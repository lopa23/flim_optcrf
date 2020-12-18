import torch
import torch.nn
import sys
print(torch.__version__)
print(torch.version.cuda)

print(sys.path)
import numpy.random as npr
from torch.autograd import Function, Variable
#from pyro import Parameter
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.multiprocessing as mp
import pandas
import scipy.io
import os
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

#from scipy.linalg import block_diag
import tables
import argparse
import matplotlib.image as mpimg

from skimage.transform import rescale, resize
import math

from torch.utils.data import DataLoader
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
from matplotlib import cm

from skimage.transform import rescale, resize
sys.path.append('../pytorch-crfasrnn-master/Permutohedral_Filtering/')
from Permutohedral_Filtering import PermutohedralLayer

print(torch.cuda.is_available())
x=torch.tensor(1.0).cuda()
y=torch.tensor(2.0).cuda()
print(x+y)

x=x.cpu().numpy()
scipy.io.savemat('a1.mat',{'x':x})
