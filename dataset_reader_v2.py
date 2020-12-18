import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from skimage.transform import rescale, resize
import scipy.io
import h5py
import tables
from torch.utils.data import DataLoader

def kron(matrix1, matrix2):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    r=matrix1.size(0)
    R=repeat_along_diag(matrix2,r)
    
    #R=torch.zeros(n*m,n*m)

    return R
   
def rgb2gray(rgb):
    r, g, b=rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray= .2989*r + .5870*g+ .114*b
    return gray
   
def repeat_along_diag(a, r):
    m,n = a.shape
   
    out = np.zeros((r,m,r,n), dtype=np.float32)
    diag = np.einsum('ijik->ijk',out)
    diag[:] = (a)
    return out.reshape(-1,n*r)

def read_mat_file(fname):
   
    file = tables.open_file(fname)
    Q = file.root.HH[:]
    p=file.root.f[:]
    G=file.root.D[:]
    m=file.root.m[:]
   

    m=int(m[0].item())
    
    Q=torch.tensor(Q).float();
    E=torch.eye(m)
    #Q=torch.from_numpy(kron(E,Q)).float()
    print("Size m, Q",m, Q.size())
    
    n=Q.size(0)
    p=torch.tensor(p).float();
    p=p.t()
    p1=p.size(0)

    G=torch.tensor(G).float();
    if(p1==1):
       G=G.t()
    
    gx=G.size(0)
    gy=G.size(1)
    
    h = torch.tensor(np.zeros((gx, 1))).float();
    
    
    temp=np.zeros((1,n))
    temp[0,n-1]=.000000001

    A = torch.from_numpy(temp);
    b = torch.from_numpy(np.zeros((1,1)));
    return Q, p, G, h, A, b, m

class MyDataset(Dataset):
    def __init__(self, data_root):
        self.samples = []
        self.data_root=data_root
        self.train_folder=[];
        idx=0
        
        for folname in os.listdir(data_root):
            
            self.train_folder.append(os.path.join(self.data_root, folname))
            print(self.train_folder[0])
            subfolnames=os.listdir(self.train_folder[idx]);
            idx1=0
            # this is to avoid reading the output folder
            for folname in subfolnames:
                  if folname !='output':
                      subfolnames[idx1]=folname
                      idx1=idx1+1
                      
            subfol_path1=os.path.join(self.train_folder[idx],subfolnames[0]);
            subfol_path2=os.path.join(self.train_folder[idx],subfolnames[1]);
            print(subfol_path1,' ',subfol_path2)
            #reading 1st modality
            for thisfile in os.listdir(subfol_path1):
                
                this_filepath = os.path.join(subfol_path1, thisfile)
            
                if(this_filepath.find('image.bmp')!=-1):
                    img= mpimg.imread(this_filepath);
                    if(img.ndim >2):
                        img=rgb2gray(img)
                    img=img.astype(np.float32)
                    img=torch.from_numpy(img)
                    #img=img.unsqueeze(0)
                    
                elif(this_filepath.find('truth.bmp')!=-1):
                    target= torch.from_numpy(mpimg.imread(this_filepath))
                   
                elif(this_filepath.find('.txt')!=-1):
                   
                    label = np.loadtxt(this_filepath, dtype='i', delimiter=',')
                    n1, n2=label.shape
                    if(n2>1):
                        Pixel_pos1=torch.from_numpy(label[:,[0, 1]])
                        Pixel_pos1=Pixel_pos1.type(torch.uint8)
                        anno1=torch.from_numpy(label[:,2])
                        
                    else:
                        Pixel_pos1=None
                        anno1=torch.from_numpy(label)
                elif(this_filepath.find('.mat')!=-1):
                   
                    Q1, p1, G1, h1, A1, b1, m1=read_mat_file(this_filepath)
                   
            #reading 2nd modality
            for thisfile in os.listdir(subfol_path2):
                
                this_filepath = os.path.join(subfol_path2, thisfile)
            
                if(this_filepath.find('.txt')!=-1):
                   
                    label = np.loadtxt(this_filepath, dtype='i', delimiter=',')
                    n1, n2=label.shape
                    if(n2>1):
                        Pixel_pos2=torch.from_numpy(label[:,[0, 1]])
                        Pixel_pos2=Pixel_pos2.type(torch.uint8)
                        anno2=torch.from_numpy(label[:,2])
                    
                    else:
                        Pixel_pos2=None
                        anno2=torch.from_numpy(label)
                elif(this_filepath.find('.mat')!=-1):
                    Q2, p2, G2, h2, A2, b2, m2=read_mat_file(this_filepath)
                
            idx=idx+1
           
            item=(img, target, anno1, Pixel_pos1, Q1, p1, G1, h1, m1, anno2, Pixel_pos2, Q2, p2, G2, h2, m2)
           
            self.samples.append(item)
            #self.samples.append({'image': img, 'target': target, 'Anno':anno, 'Pixel_pos':Pixel_pos, 'Q':Q, 'p':p, 'G':G, 'h':h, 'm':m})
        
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return idx, self.samples[idx]
    


