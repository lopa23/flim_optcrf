import faulthandler
from qpth.qp import QPFunction
import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
from torch.autograd import Function, Variable
#from pyro import Parameter
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.multiprocessing as mp
import pandas
import scipy.io
import os
import sys
sys.path.remove('/usr/local/lib/python3.6/dist-packages/')
print(sys.path)
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

from CRFasRNN import CRFasRNN as crfrnn
from dataset_reader import MyDataset
from torch.utils.data import DataLoader
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
from matplotlib import cm
from Kmeansnet import kmeans

def my_collate(batch):
    
    for this_batch in batch:
        item = this_batch[1]
        img=item[0]
        target=item[1]
        
        anno1 = item[2]
        Pixel_pos1 = item[3]
        Q1 = item[4]
        p1 = item[5]
        G1 = item[6]
        h1 = item[7]
        m1 = item[8]

        anno2 = item[9]
        Pixel_pos2 = item[10]
        Q2 = item[11]
        p2 = item[12]
        G2 = item[13]
        h2 = item[14]
        m2 = item[15]
   
    return [img, target, anno1, Pixel_pos1, Q1, p1, G1, h1, m1, anno2, Pixel_pos2, Q2, p2, G2, h2, m2]

def modify_output(output,Pixel_pos,nsize):
    n1, n2=output.size()
    p1, p2=Pixel_pos.shape
    
    new_output=torch.zeros(n1, nsize, nsize,dtype=torch.float32);
    
    for k in range(0,p1):
        x, y =Pixel_pos[k,[0, 1]]
        x=x.item()
        y=y.item()
        #print(x,y,output[:,k])
        new_output[:,x-1,y-1]=output[:,k]
       
    return new_output

class Ensemble(nn.Module):
     def __init__(self,optNet,km):
        super(Ensemble, self).__init__()
        device=torch.device("cuda:7")
        network=torch.rand(1,1,10,10)#random network
        self.opt1 = OptNet(1, 1, 3, 4)#.cuda(0)
        
        self.opt1.to(device)
        self.opt2 = OptNet(1, 1, 3, 4)#.cuda(1)
        
        self.opt2.to(device)
        #self.opt.cuda()
        self.km = kmeans(2)#.cuda()

     def process_opt(self, opt, x, Q, p, G, h, m, img, anno, Pixel_pos):
        n=img.size()
        if(m<5000):
            output=opt(x,Q,p,G,h,m).float()#.cuda(0)#.to(device)
        
            output=-1*output.view(10,-1)
       
            output=output/torch.max(output)
        else:
            o=np.random.rand(10,m*10)
            o =np.array(o).astype(np.single)
            output=torch.from_numpy(o).cuda()
        if(Pixel_pos is None):
            output=output.view(-1,n[1],n[2]);
        else:
            output=modify_output(output,Pixel_pos,n[1])
        
        output=output.type(torch.float32)
                
        return output
        
     def forward(self, x, Q1, p1,G1,h1,m1,img, anno1,Pixel_pos1, Q2, p2, G2,h2,m2,anno2,Pixel_pos2):

        output1=self.process_opt(self.opt1, x, Q1, p1, G1, h1, m1, img, anno1, Pixel_pos1)
        output2=self.process_opt(self.opt2, x, Q2, p2, G2, h2, m2, img, anno2, Pixel_pos2)
        output=torch.cat((output1,output2),0).float()
        #print("size at this point1",output.size())
        output_k=output.view(20,-1)
        output_k=torch.transpose(output_k,0,1)
        #print("size at this point2-1",output_k.size())
        output_k=output_k.cpu().detach().numpy()[:,:]
        #print("size at this point2",output_k.shape)
        
        img=img.float()
        
        anno1=anno1.float()
        #y=self.crf(img, anno1,Pixel_pos1)
        self.km.kmeanstrain(output_k)
        y=self.km.kmeansfwd(output_k)
        
        y=torch.from_numpy(y)
        y=y.squeeze(0)
        #y=y[0,:,:] #need to look into this
        print('Final Output',y.size())
        #p1, p2=y.size()
        
        return y, output

    #def main(self):
        #self.init()
class OptNet(nn.Module):
    def __init__(self, nFeatures, nHidden, nCls, bn, nineq=1, neq=0, eps=1e-4):
        super(OptNet,self).__init__()
        self.device = torch.device("cuda:4")
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
        
        self.M =Variable(X,requires_grad=True)
        self.L = Parameter(torch.tril(torch.rand(nCls, nCls)),requires_grad=True)
        self.p= Parameter(torch.Tensor(1,nCls).uniform_(-1,1),requires_grad=True)
        self.G =Parameter(torch.Tensor(nineq,nCls).uniform_(-1,1),requires_grad=True)
        #self.A =Parameter(torch.Tensor(neq,nCls).uniform_(-1,1).cuda())
        #self.b=
        self.z0 = Parameter(torch.zeros(nCls),requires_grad=True)
        self.s0 =Parameter(torch.ones(nineq),requires_grad=True)

    def forward(self, x, Q, p, G, h,m):
        print("Cuda current device", torch.cuda.current_device())  
        nBatch = x.size(0)
       
        if(m>1):
            p=p.float().t()
        else:
            p=p.float()

        G=G.float()#.cuda()
        Q=Q.float()
        if(m>=2):
          Q=Q.unsqueeze(0)
        h=h.float()
        #print(Q.size(),p.size(),G.size(),h.size())
        
        e = Variable(torch.Tensor(),requires_grad=True)
       
        x = QPFunction(verbose=True)(Q, p, G, h, e, e)#.cuda()
      
        x=x.view(10,-1) ##this was not needed earlier
        
        return F.log_softmax(x,dim=1)
    
    
def train(net,train_loader,traindataset,data_root_train):

    follist_train= os.listdir(data_root_train)
    for param in net.opt1.parameters():
            param.requires_grad = False
    for param in net.opt2.parameters():
            param.requires_grad = False
    #for param in net.crf.parameters():
            #param.requires_grad = False
            
    x=torch.tensor(([1., 2., 3., 4., 5., 6.]))
    
    device = torch.device("cuda:7")
    
    optimizer = torch.optim.SGD([
                {'params':list(net.opt1.parameters())+list(net.opt2.parameters()), 'lr': 1e-3}])

    
    y=torch.rand(1,10)
    print("Number of training data",len(train_loader.dataset))
    it=iter(train_loader)
    for i in range(0,len(train_loader.dataset)):
        fol_name=os.path.join(data_root_train, follist_train[i])
        print(fol_name)
        if not os.path.exists(os.path.join(fol_name,'output/')):
            os.makedirs(os.path.join(fol_name,'output/'))
                              
        filename_save=os.path.join(fol_name,'output/output.png')
        filename_save_opt=os.path.join(fol_name,'output/opt_output.mat')
        optimizer.zero_grad()
        (img, target, anno1, Pixel_pos1, Q1, p1, G1, h1, m1, anno2, Pixel_pos2, Q2, p2, G2, h2, m2)=next(it)
       
        target=target.squeeze(0)
        anno1=anno1.squeeze(0)
        Pixel_pos1=Pixel_pos1.squeeze(0)
        Q1=Q1.squeeze(0)
       
        p1=p1.squeeze(0)
        G1=G1.squeeze(0)
        h1=h1.squeeze(0)
        anno2=anno2.squeeze(0)
        Pixel_pos2=Pixel_pos2.squeeze(0)
        Q2=Q2.squeeze(0)
        p2=p2.squeeze(0)
        G2=G2.squeeze(0)
        h2=h2.squeeze(0)
    
        y, opt_output=net(x,Q1,p1,G1,h1,m1,img, anno1,Pixel_pos1,Q2,p2,G2,h2,m2,anno2,Pixel_pos2)
        #writing both outputs to files
        y1=y.detach().cpu()
        opt_output1=opt_output.detach().numpy()
        scipy.io.savemat(filename_save_opt,{'opt_out':opt_output1})
        write_output_image(y1,filename_save,False)

        #calculating losses
        target=target.type(torch.double)
        criterion=nn.BCELoss()
        loss=criterion(y1,target)
        #loss = F.mse_loss(y, target)
        loss = Variable(loss, requires_grad = True)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        print("Loss value",loss.item())
        loss.backward()
        optimizer.step()
        """del(Q1)
        del(G1)
        del(Q2)
        del(G2)
        del(p1)
        del(p2)
        del(h1)
        del(h2)
        del(anno1)
        del(anno2)
        del(Pixel_pos1)
        del(Pixel_pos2)
        torch.cuda.empty_cache()
        """
    return y

def test(net,test_loader,data_root_test):
    
    follist_test= os.listdir(data_root_test)
    x=torch.rand(6);
    net.eval()
    print("Here in test, data read")
    it=iter(test_loader)

   
    for i in range(0,len(test_loader.dataset)):
        fol_name=os.path.join(data_root_test, follist_test[i])
        print(fol_name)
        if not os.path.exists(os.path.join(fol_name,'output/')):
            os.makedirs(os.path.join(fol_name,'output/'))
                              
        filename_save=os.path.join(fol_name,'output/output.png')
       
        (img, target, anno1, Pixel_pos1, Q1, p1, G1, h1, m1, anno2, Pixel_pos2, Q2, p2, G2, h2, m2)=next(it)
       
        target=target.squeeze(0)
        anno1=anno1.squeeze(0)
        Pixel_pos1=Pixel_pos1.squeeze(0)
        Q1=Q1.squeeze(0)
       
        p1=p1.squeeze(0)
        G1=G1.squeeze(0)
        h1=h1.squeeze(0)
        anno2=anno2.squeeze(0)
        Pixel_pos2=Pixel_pos2.squeeze(0)
        Q2=Q2.squeeze(0)
        p2=p2.squeeze(0)
        G2=G2.squeeze(0)
        h2=h2.squeeze(0)

    
    
        print("Here in test, data read", fol_name)
    
   
        output = net(x,Q1,p1,G1,h1,m1,img, anno1, Pixel_pos1,Q2,p2,G2,h2,m2,anno2,Pixel_pos2)
      
        write_output_image(output,filename_save,False)
        ##add code to test the output wrt to target
        """ del(Q1)
        del(G1)
        del(Q2)
        del(G2)
        del(p1)
        del(p2)
        del(h1)
        del(h2)
        del(anno1)
        del(anno2)
        del(Pixel_pos1)
        del(Pixel_pos2)
        torch.cuda.empty_cache()
       """ 
    return output

def  write_output_image(y,filename_save,show):
    selem = disk(2)
    print(y)
    if(len(y)>1):
       y=y[0]
      
    y1=y.cpu().detach().numpy()
    #print("Before resize",y1.shape)
    y1=np.reshape(y1,(256,256))
    #print("After resize",y1.shape)
    #y1=closing(y1,selem)
    if (show):
        plt.imshow(y1, interpolation='nearest')
        plt.show()
    plt.imsave(filename_save,np.uint8(y1),cmap = cm.gray)
    
def main():
    #faulthandler.enable()
    #os.environ['CUDA_VISIBLE_DEVICES']='1,2,3,4,5'
    torch.cuda.set_device(7)
    #mp.set_start_method("spawn")

    #print(torch.cuda.current_device())
     ##Setting up parameters
    parser = argparse.ArgumentParser(description='PyTorch Regression-classifcation Example')

    parser.add_argument('--batchSize', type=int, default=10, help='Training batch size')
    parser.add_argument('--testBatchSize', type=int, default=10, help='Testing batch size')
    parser.add_argument('--nEpochs', type=int, default=2, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
    parser.add_argument('--cuda', action='store_true', help='Use cuda?')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='Random seed to use. Default=123')
    
    nEpoch=4
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str)
    args = parser.parse_args()
    
    if args.save is None:
        t = '{}.{}'.format('mydata', 'optnet')
    args.save = os.path.join('work', t)

    data_root_train='../tissue_data_2mod/train/'
    data_root_test='../tissue_data_2mod/test/'
    TR=False # will perform training
    TT=True # will perform testinf
   ##File read and image reads,this needs to be converted to batch mode later on
    net=Ensemble(OptNet,kmeans)#.cuda()
    if(TR):
       
        traindataset = MyDataset((data_root_train))
        train_loader = DataLoader(traindataset, batch_size=1,  collate_fn=my_collate)#pin_memory=True, num_workers=0,
        for epoch in range(1, nEpoch + 1):
        
            print("Epoch ", epoch)
    
            y=train(net,train_loader,traindataset,data_root_train)
            
            fname='saved_model/net'+str(epoch)+'.pth'
            torch.save(net.state_dict(),fname)
            
        
        
        
    if(TT):
        testdataset = MyDataset((data_root_test))

        test_loader = DataLoader(testdataset, batch_size=1,num_workers=0,collate_fn=my_collate)
        net.load_state_dict(torch.load('saved_model/net4.pth'))
        y=test(net,test_loader,data_root_test)
       
if __name__ == '__main__':
    main()

    



    
