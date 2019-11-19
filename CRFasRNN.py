"""
        This function creates an CRF as RNN model. See this https://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf for
        details.
        It depends on https://github.com/sadeepj/crfasrnn_keras/ and https://github.com/MiguelMonteiro/CRFasRNNLayer
"""
import torch.nn as nn
import numpy as np
#from Network.Blocks.Permutohedral_Filter

import configparser
import torch
import sys
sys.path.insert(0, '../pytorch-crfasrnn-master/Permutohedral_Filtering/')
from Permutohedral_Filtering import PermutohedralLayer
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from skimage.transform import rescale, resize
import math

class CRFasRNN(nn.Module):

    def __init__(
        self,
        network,
        nb_iterations,
        nb_classes,
        nb_input_channels,
        theta_alpha,
        theta_beta,
        theta_gamma,
        gpu_rnn
    ):
        """

        The crf as rnn model combines a neural network with a crf model. The crf can be written as recurrent neural net
        work. The network delivers the unaries. The rnn uses mean field approximation to improve the results
        :param network: The network to deliver the unaries. The input must have the shape [batch, nb_input_channels,
        width, height]. The output must have the shape [batch, nb_classes, width, height]
        :param nb_iterations: Who many iterations does the rnn run. In the paper 5 are used for train and 10 for test
        :param nb_classes: Who many segments does the output have
        :param nb_input_channels: number of color channels. should be 1 or 3.
        :param theta_alpha: used for permutohedral filter. See paper for details
        :param theta_beta:used for permutohedral filter. See paper for details
        :param theta_gamma:used for permutohedral filter. See paper for details
        :param gpu_rnn: On which GPU does the RNN run. None if CPU.
        """
        super(CRFasRNN, self).__init__()
        self.use_gpu = gpu_rnn is not None
        self.gpu = gpu_rnn
        self.network = network
        self.nb_iterations = nb_iterations
        self.nb_classes = nb_classes
        self.nb_input_channels = nb_input_channels

        # These are the elements for the filtering
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma



        self.spatial_filter = PermutohedralLayer(
            bilateral=False,
            theta_alpha=self.theta_alpha,
            theta_beta=self.theta_beta,
            theta_gamma=self.theta_gamma
        )

        # This is the convolutional layer for spatial filtering
        self.spatial_conv = nn.Conv2d(
            in_channels=nb_classes,#done by me
            out_channels=nb_classes,
            kernel_size=(1, 1)
        )

        self.bilateral_filter = PermutohedralLayer(
            bilateral=True,
            theta_alpha=self.theta_alpha,
            theta_beta=self.theta_beta,
            theta_gamma=self.theta_gamma
        )

        # This is the convolutional layer for bilateral filtering
        self.bilateral_conv = nn.Conv2d(
            in_channels=nb_classes,
            out_channels=nb_classes,
            kernel_size=(1, 1)
        )

        # This is the convolutional layer for compatiblity step
        self.compatitblity_conv = nn.Conv2d(
            in_channels=nb_classes,
            out_channels=nb_classes,
            kernel_size=(1, 1)
        )



        # push all the thing to the gpu
        if self.use_gpu:
            self.bilateral_conv.cuda(gpu_rnn)
            self.compatitblity_conv.cuda(gpu_rnn)
            self.bilateral_filter.cuda(gpu_rnn)
            self.spatial_conv.cuda(gpu_rnn)
            self.spatial_filter.cuda(gpu_rnn)
            
         
        # check whether crf should be updated
        #if train_mode.lower() == "cnn":
        self.bilateral_conv.weight.requires_grad = True
        self.bilateral_conv.bias.requires_grad = True
        self.compatitblity_conv.weight.requires_grad = True
        self.compatitblity_conv.bias.requires_grad = True
        self.spatial_conv.weight.requires_grad = True
        self.spatial_conv.bias.requires_grad = True
        self.bilateral_filter.requires_grad=True
         
        # softmax
        self.softmax = nn.Softmax2d()
        self.softmax.requires_grad=True
        self.out_act = nn.Threshold(.5,0)
        self.out_act1 = nn.Threshold(-0.000001,1)
        
    def forward(
            self,
            image,
            features,
            anno,
            Pixel_pos,
            name=None
    ):

        # calculate the unaries
        
        #unaries = self.network(image)##changed by me
        #unaries=image.unsqueeze(0).cuda()
        unaries=generate_unaries(image,anno, Pixel_pos)
       
        unaries=unaries.unsqueeze(0).cuda()
        print("Image Size",image.size())
        print("Unaries Size",unaries.size())
        print("Feature Size",features.size())
        if self.use_gpu:
            unaries = unaries.cuda(self.gpu)

        # set the q_values
        feat = features.cuda(0)
        q_values = unaries
        softmax_out = self.softmax(q_values)

        #
        #self.nb_iterations=1;
        #print(q_values,softmax_out)
        for i in range(self.nb_iterations):
            print("Iteration",i)
            # 1. Filtering
            # 1.1 spatial filtering
           
            spatial_out = self.spatial_filter(
                softmax_out,
                feat
            )
            #print(self.bilateral_filter)
            # 1.2 bilateral filtering
            #bilateral_out=torch.zeros(spatial_out.size(),requires_grad=True)
            #bilateral_out.retain_grad()
            #N_softmax_out=torch.ones(softmax_out.size()).cuda()
            bilateral_out = self.bilateral_filter(
                softmax_out,
                feat
            )
            
            #print(bilateral_out.requires_grad,bilateral_out.grad_fn,bilateral_out)
            # 2. weighted filter outputs
           
            message_passing = self.spatial_conv(spatial_out) #+ self.bilateral_conv(bilateral_out)
           
            # 3. compatibilty transform
            
            pairwise = self.compatitblity_conv(message_passing)
            
            
            # 4. add pairwise terms
            q_values = unaries - pairwise

            # 5. Softmax
            softmax_out = self.softmax(q_values)
            y = self.out_act(softmax_out)
            y = self.out_act1(-1*y)
            
        return y#softmax_out

    def crf_dict(self):
        return {
            "spatial_conv": self.spatial_conv.state_dict(),
            "bilateral_conv": self.bilateral_conv.state_dict(),
            "compatitblity_conv": self.compatitblity_conv.state_dict()
        }

    def cnn_dict(self):
        return self.network.state_dict()

    def cnn_parameters(self):
        return self.network.parameters()

    def crf_parameters(self):
        return [
            self.spatial_conv.bias,
            self.spatial_conv.weight,
            self.bilateral_conv.bias,
            self.bilateral_conv.weight,
            self.compatitblity_conv.bias,
            self.compatitblity_conv.weight
        ]

    def load_parameter(
            self,
            cnn_path=None,
            crf_path=None
    ):
        if cnn_path is not None:
            self.network.load_state_dict(torch.load(cnn_path))
        if crf_path is not None:
            state_dict = torch.load(crf_path)
            self.spatial_conv.load_state_dict(state_dict["spatial_conv"])
            self.bilateral_conv.load_state_dict(state_dict["bilateral_conv"])
            self.compatitblity_conv.load_state_dict(state_dict["compatitblity_conv"])

def generate_unaries(img,anno,Pixel_pos):
    GT_PROB = 0.5
    if(Pixel_pos is None):
        n=img.shape
    else:
        n=[1, 256, 256]
    
    M=3;
    num_pixel=n[1];
    
    u_energy = -math.log(1.0/ M);
    n_energy = -math.log((1.0 - GT_PROB)/(M-1));
    p_energy = -math.log(GT_PROB);
    label=[10, 120];
    unaries=(torch.zeros(2,n[1],n[2]));
    #unaries[1,:,:]=torch.ones(1,n[1],n[2]);
    if(Pixel_pos is None):
        for i  in range(0,n[1]):
            for j in range(0,n[2]):
                if(anno[i,j]==0):
                    unaries[0,i,j]=p_energy - math.log(math.exp(-pow(img[0,i,j]-label[0],2)/pow(100,2)))
                    unaries[1,i,j]=n_energy- math.log(math.exp(-pow(img[0,i,j]-label[1],2)/pow(100,2)))
                elif(anno[i,j]==1):
                    unaries[1,i,j]=p_energy- math.log(math.exp(-pow(img[0,i,j]-label[1],2)/pow(100,2)))
                    unaries[0,i,j]=n_energy- math.log(math.exp(-pow(img[0,i,j]-label[0],2)/pow(100,2)))
    else:
        p1, p2=Pixel_pos.shape
        
        for k in range(0,p1):
            x, y =Pixel_pos[k,[0, 1]]
            x=x.item()
            y=y.item()
            #print(k,x,y,anno[k])
            if(anno[k]==0):
                unaries[0,x-1,y-1]=p_energy - math.log(math.exp(-pow(img[0,x-1,y-1]-label[0],2)/pow(100,2)))
                unaries[1,x-1,y-1]=n_energy- math.log(math.exp(-pow(img[0,x-1,y-1]-label[1],2)/pow(100,2)))
               # print(unaries[[0,1],x-1,y-1],img[0,x-1,y-1])
            elif(anno[k]==1):
                unaries[1,x-1,y-1]=p_energy- math.log(math.exp(-pow(img[0,x-1,y-1]-label[1],2)/pow(100,2)))
                unaries[0,x-1,y-1]=n_energy- math.log(math.exp(-pow(img[0,x-1,y-1]-label[0],2)/pow(100,2)))
                #print(unaries[[0,1],x,y],img[0,x-1,y-1])
            
    #unaries[1,:,:]=20-img[0,:,:]#1 for images in 0-1 range
    return unaries
    
