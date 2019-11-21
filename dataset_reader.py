import os
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
   
    
def repeat_along_diag(a, r):
    m,n = a.shape
    out = np.zeros((r,m,r,n), dtype=float)
    diag = np.einsum('ijik->ijk',out)
    diag[:] = (a)
    return out.reshape(-1,n*r)

def read_mat_file():
    fname='../mycode/H.mat'
    file = tables.open_file(fname)
    Q = file.root.HH[:]
    p=file.root.f[:]
    G=file.root.D[:]
    m=file.root.m[:]
   

    m=int(m[0].item())
    
    Q=torch.tensor(Q).float();
    E=torch.eye(m)
    Q=torch.from_numpy(kron(E,Q))
    print("Size m, Q",m, Q.size())
    
    n=Q.size(0)
    p=torch.tensor(p);
    p=p.t()
    p1=p.size(0)

    G=torch.tensor(G);
    if(p1==1):
       G=G.t()
    
    gx=G.size(0)
    gy=G.size(1)
    
    h = torch.tensor(np.zeros((gx, 1)));
    
    
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
            print(self.train_folder[idx])
            for thisfile in os.listdir(self.train_folder[idx]):
                this_filepath = os.path.join(self.train_folder[idx], thisfile)
            
                if(this_filepath.find('image.bmp')!=-1):
                    img= mpimg.imread(this_filepath);
                    img=torch.from_numpy(img)
                    #img=img.unsqueeze(0)
                    
                elif(this_filepath.find('truth.bmp')!=-1):
                    target= mpimg.imread(this_filepath)
                   
                elif(this_filepath.find('.txt')!=-1):
                   
                    label = np.loadtxt(this_filepath, dtype='i', delimiter=',')
                    p1, p2=label.shape
                    if(p2>1):
                        Pixel_pos=torch.from_numpy(label[:,[0, 1]])
                        Pixel_pos=Pixel_pos.type(torch.uint8).cuda()
                        anno=torch.from_numpy(label[:,2])
                        
                    else:
                        Pixel_pos=None
                        anno=torch.from_numpy(label)
                elif(this_filepath.find('.mat')!=-1):
                    Q, p, G, h, A, b, m=read_mat_file()
          
            self.samples.append({'image': img, 'target': target, 'Anno':anno, 'Pixel_pos':Pixel_pos, 'Q':Q, 'p':p, 'G':G, 'h':h, 'm':m})
            
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return idx, self.samples[idx]

def main():
    dataset = MyDataset('../0ng/export_modified/train/')
    print(len(dataset))
    s1=dataset[0]
    print(s1['image'].shape)
    
#main()


