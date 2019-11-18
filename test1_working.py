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
#from mat4py import loadmat

def main():
    print(torch._C._cuda_getDriverVersion())
    print(torch.__version__)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    #Q, p, G, h, A, b=read_mat_file()
    
    opt=OptNet(1, 1, 5, 4)
    x=torch.tensor(([1., 2., 3., 4., 5., 6.]))
    #x=x.unsqueeze(0)
    #x=x.cuda()
    device = torch.device("cuda")
    y=opt(x).to(device)
    #print(y)
    #opt.forward()
    
def read_mat_file():
    data = scipy.io.loadmat('../boxqp-master/H.mat');

    #print(data)
    Q=torch.tensor(data["HH"]);

    n=Q.size(0)
    p=torch.tensor(data["f"]);
    p=p.t()
    m=p.size(0)

    G=torch.tensor(data["D"]);
    G=G.t()
    gx=G.size(0)
    gy=G.size(1)

    

    h = torch.tensor(np.zeros((gx, 1)));
    
    
    temp=np.zeros((1,n))
    temp[0,n-1]=.000000001

    A = torch.from_numpy(temp);
    b = torch.from_numpy(np.zeros((1,1)));
    return Q, p, G, h, A, b

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

        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden)
            self.bn2 = nn.BatchNorm1d(nCls)

        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, nCls)
        X=torch.tril(torch.ones(nCls, nCls))
        
        self.M =Variable(X.cuda())
        self.L = Parameter(torch.tril(torch.rand(nCls, nCls).cuda()))
        self.p= Parameter(torch.Tensor(1,nCls).uniform_(-1,1).cuda())
        self.G =Parameter(torch.Tensor(nineq,nCls).uniform_(-1,1).cuda())
        #self.A =Parameter(torch.Tensor(neq,nCls).uniform_(-1,1).cuda())
        #self.b=
        self.z0 = Parameter(torch.zeros(nCls).cuda())
        self.s0 =Parameter(torch.ones(nineq).cuda())

    def forward(self, x):
        nBatch = x.size(0)
       
        # FC-ReLU-(BN)-FC-ReLU-(BN)-QP-Softmax
        x = x.view(nBatch, -1)
        
        x=x.unsqueeze(0)
        
       
        x=x.float()
        tmp=self.fc1(x)
       
        x = F.relu(tmp)

        x=x.squeeze(2)
       
        #if self.bn:
            #x = self.bn1(x)
        #x = F.relu(self.fc2(x))
        #if self.bn:
            #x = self.bn2(x)

        L = self.M*self.L
        Q = L.mm(L.t()) + self.eps*Variable(torch.eye(self.nCls)).cuda()
        p=self.p.double()
        h = self.G.mv(self.z0)+self.s0
        G=self.G.double()
        Q=Q.double()
        h=h.double()
        print(Q.size(),p.size(),G.size(),h.size())
        
        e = Variable(torch.Tensor())
       
        x = QPFunction(verbose=True)(Q, p, G, h, e, e).cuda()
        print(x)
        return F.log_softmax(x,dim=1)
    
  
        
if __name__ == '__main__':
    main()
    
