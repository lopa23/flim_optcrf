from qpth.qp import QPFunction
import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
from torch.autograd import Function, Variable
#from pyro import Parameter
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import pandas
import scipy.io
import h5py
import os
#from scipy.linalg import block_diag
import tables
import argparse
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from skimage.transform import rescale, resize
import math
#from . import CRFasRNN as crf
import sys
sys.path.insert(0, '.')
from  CRFasRNN import CRFasRNN as crfrnn


def modify_output(output,Pixel_pos,nsize):
    n1, n2=output.size()
    p1, p2=Pixel_pos.shape
    new_output=torch.zeros(n1, nsize, nsize);
    for k in range(0,p1):
        x, y =Pixel_pos[k,[0, 1]]
        x=x.item()
        y=y.item()
        new_output[:,x-1,y-1]=output[:,k]

    return new_output


def get_loaders():
    Q, p, G, h, A, b, m=read_mat_file()
    img= mpimg.imread('../0ng/export_modified/00ng_im3_NADH_PC102_60s_40x_oil_intensity_image.bmp');
    img=torch.from_numpy(img)
    img=img.unsqueeze(0)
    target=mpimg.imread('../0ng/export_modified/00ng_im3_NADH_PC102_60s_40x_oil_truth.bmp');
    #print('target',target.shape)
  
    #label = np.loadtxt("../mycode/label_00ng_im3_NADH_PC102_60s_40x.txt", dtype='i', delimiter=',')
    label = np.loadtxt("../0ng/export_modified/label.txt", dtype='i', delimiter=',')
    p1, p2=label.shape
    if(p2>1):
        Pixel_pos=torch.from_numpy(label[:,[0, 1]])
        Pixel_pos=Pixel_pos.type(torch.uint8).cuda()
        #Pixel_pos=None
        anno=torch.from_numpy(label[:,2])
        #anno=torch.round(torch.rand(256,256))
    else:
        Pixel_pos=None
        anno=torch.from_numpy(label)
    
    #print(anno.size(), Pixel_pos.size())
    return Q, p, G, h, A, b, m, img, anno, Pixel_pos, target

def main():
     ##Setting up parameters
    parser = argparse.ArgumentParser(description='PyTorch Regression-classifcation Example')
    parser.add_argument('--upscale_factor', type=int, default=8, required=True, help="Super resolution upscale factor")
    parser.add_argument('--full_size', type=int, default=1024, required=True, help="Size of target image")
    parser.add_argument('--batchSize', type=int, default=10, help='Training batch size')
    parser.add_argument('--testBatchSize', type=int, default=10, help='Testing batch size')
    parser.add_argument('--nEpochs', type=int, default=2, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
    parser.add_argument('--cuda', action='store_true', help='Use cuda?')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='Random seed to use. Default=123')
    
    nEpoch=1
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str)
    args = parser.parse_args()
    
    if args.save is None:
        t = '{}.{}'.format('mydata', 'optnet')
    args.save = os.path.join('work', t)
    
   ##File read and image reads,this needs to be converted to batch mode later on
    Q, p, G, h, A, b, m, img, anno, Pixel_pos, target = get_loaders()

    ##end of file reads

 
        
    
    net=Ensemble(OptNet,crfrnn)
    for epoch in range(1, nEpoch + 1):
        y=train(net,Q, p, G, h, m, img, anno.detach().numpy(),Pixel_pos,target)
   #     test(epoch,net,Q, p, G, h, m, img, anno.detach().numpy())
    #    try:
    #        torch.save(net, os.path.join(args.save, 'latest.pth'))
     #   except:
      #      pass
    y=y.cpu().detach().numpy()
    plt.imshow(y, interpolation='nearest')
    plt.show()

def train(net, Q, p, G, h, m, img, anno, Pixel_pos, target):
    
    #print('target',target.shape)
     
    for param in net.opt.parameters():
            param.requires_grad = False
    for param in net.crf.parameters():
            param.requires_grad = False
            
    x=torch.tensor(([1., 2., 3., 4., 5., 6.]))
    
    device = torch.device("cuda")
    
    optimizer = torch.optim.Adam([
                {'params':list(net.opt.parameters())+list(net.crf.parameters()), 'lr': 1e-3}])

    optimizer.zero_grad()

    y=net(x,Q,p,G,h,m,img, anno,Pixel_pos)
    #next few lines is to make loss function work (change later)
    #output=output.view(p2/2,2)
    #output=torch.rand(p2,2).cuda() #this should be of size n x c
    #target=torch.rand(y.size())
    target=torch.from_numpy(target) #this will be fixed later, should be of size n
    
    target=target.type(torch.float).cuda()
    
    loss = F.mse_loss(y, target)
    loss = Variable(loss, requires_grad = True)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
    print("Loss value",loss.item())
    loss.backward()
    optimizer.step()

    return y

def test(epoch, net, Q, p, G, h, m, img, anno):
    x=torch.rand(6);
    net.eval()
    target=torch.from_numpy(anno)
    
    target = Variable(target)
    output = net(x,Q,p,G,h,m,img, anno)
    ##add code to test the output wrt to target
    print(output)
    print(target)


class Ensemble(nn.Module):
     def __init__(self,optNet,crfrnn):
        super(Ensemble, self).__init__()
        network=torch.rand(1,1,10,10)#random network
        self.opt = OptNet(1, 1, 6, 4)
        self.crf = crfrnn(network, 10, 2 , 5, .1, .1, .1, None).cuda()

     def forward(self, x,Q,p,G,h,m,img, anno, Pixel_pos):
        n=img.size()                   
        output=self.opt(x,Q,p,G,h,m)#.to(device)
        #output=torch.rand(10,6186);
        if(Pixel_pos is None):
            output=output.view(-1,n[1],n[2]);
        else:
            output=modify_output(output,Pixel_pos,n[1])
        #output=torch.ones(10,256,256);     
    
        print(m,img.size(),output.size())
        y=self.crf(img, output, anno, Pixel_pos).cuda()
        y=y.squeeze(0)
        y=y[0,:,:] #need to look into this
        print('Final Output',y.size())
        p1, p2=y.size()
        return y

    #def main(self):
        #self.init()
    
class OptNet(nn.Module):
    def __init__(self, nFeatures, nHidden, nCls, bn, nineq=1, neq=0, eps=1e-4):
        super(OptNet,self).__init__()
        self.device = torch.device("cuda")
        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.bn = bn
        self.nCls = nCls
        self.nineq = nineq
        self.neq = neq
        self.eps = eps
        #self.requires_grad = False
        
            
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden)
            self.bn2 = nn.BatchNorm1d(nCls)

        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, nCls)
        X=torch.tril(torch.ones(nCls, nCls))
        
        self.M =Variable(X.cuda(),requires_grad=True)
        self.L = Parameter(torch.tril(torch.rand(nCls, nCls).cuda()),requires_grad=True)
        self.p= Parameter(torch.Tensor(1,nCls).uniform_(-1,1).cuda(),requires_grad=True)
        self.G =Parameter(torch.Tensor(nineq,nCls).uniform_(-1,1).cuda(),requires_grad=True)
        #self.A =Parameter(torch.Tensor(neq,nCls).uniform_(-1,1).cuda())
        #self.b=
        self.z0 = Parameter(torch.zeros(nCls).cuda(),requires_grad=True)
        self.s0 =Parameter(torch.ones(nineq).cuda(),requires_grad=True)

    def forward(self, x, Q, p, G, h,m):
        nBatch = x.size(0)
       
        # FC-ReLU-(BN)-FC-ReLU-(BN)-QP-Softmax
        x = x.view(nBatch, -1)
        
        x=x.unsqueeze(0)
        
       
        x=x.float()
        tmp=self.fc1(x)
       
        x = F.relu(tmp)

        x=x.squeeze(2)
       
        L = self.M*self.L
        if(m>1):
            p=p.double().t()
        else:
            p=p.double()

        G=G.double()#.cuda()
        Q=Q.double()
        if(m>=2):
          Q=Q.unsqueeze(0)
        h=h.double()
        print(Q.size(),p.size(),G.size(),h.size())
        
        e = Variable(torch.Tensor(),requires_grad=True)
       
        x = QPFunction(verbose=True)(Q, p, G, h, e, e).cuda()
        #print(x.size(),x)
        #scipy.io.savemat('../mycode/solution_qpth.mat',mdict={'x':np.array(x.cpu())})
        return F.log_softmax(x,dim=1)
    
    def main(self):
        self.init()

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
    print("Size m",m)
    Q=torch.tensor(Q).float();
    E=torch.eye(m)
    Q=torch.from_numpy(kron(E,Q))
    print("Size Q",Q.size())
    
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

if __name__ == '__main__':
    main()
    
