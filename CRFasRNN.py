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
sys.path.append('../pytorch-crfasrnn-master/Permutohedral_Filtering/')
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
        self.out_act = nn.Threshold(.5,0)


        # push all the thing to the gpu
        if self.use_gpu:
            self.bilateral_conv.cuda(gpu_rnn)
            self.compatitblity_conv.cuda(gpu_rnn)
            self.bilateral_filter.cuda(gpu_rnn)
            self.spatial_conv.cuda(gpu_rnn)
            self.spatial_filter.cuda(gpu_rnn)
            
         
        # check whether crf should be updated
        #if train_mode.lower() == "cnn":
           # self.bilateral_conv.weight.requires_grad = False
           # self.bilateral_conv.bias.requires_grad = False
            #self.compatitblity_conv.weight.requires_grad = False
            #self.compatitblity_conv.bias.requires_grad = False
            #self.spatial_conv.weight.requires_grad = False
            #self.spatial_conv.bias.requires_grad = False

        # softmax
        self.softmax = nn.Softmax2d()

    def forward(
            self,
            image,
            output,
            anno,
            Pixel_pos,
            name=None
    ):

        # calculate the unaries
        
        #unaries = self.network(image)##changed by me
        #unaries=image.unsqueeze(0).cuda()
        image=image.unsqueeze(0)
        unaries=generate_unaries(image,anno,Pixel_pos)
        
        image=output
        unaries=unaries.unsqueeze(0).cuda()
        print("Image Size",image.size())
        print("Unaries Size",unaries.size())
        if self.use_gpu:
            unaries = unaries.cuda(self.gpu)

        # set the q_values
        image = image.cuda()
        q_values = unaries
        softmax_out = self.softmax(q_values)
        for i in range(self.nb_iterations):
            print("Iteration",i)
            # 1. Filtering
            # 1.1 spatial filtering
            spatial_out = self.spatial_filter(
                softmax_out,
                image
            )
            # 1.2 bilateral filtering
            bilateral_out = self.bilateral_filter(
                softmax_out,
                image
            )
            print(bilateral_out)
            # 2. weighted filter outputs
            
            message_passing = self.spatial_conv(spatial_out) + self.bilateral_conv(bilateral_out)

            # 3. compatibilty transform
            
            pairwise = self.compatitblity_conv(message_passing)
            #print(message_passing.size(),compatibility_matrix.size(),pairwise.size())
            
            # 4. add pairwise terms
            q_values = unaries - pairwise

            # 5. Softmax
            softmax_out = self.softmax(q_values)

        softmax_out[torch.isnan(softmax_out)]=0
        y = self.out_act(softmax_out)
        
        return y

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
    unaries=torch.zeros([2,n[1],n[2]],dtype=torch.float32);
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
                unaries[0,x-1,y-1]=10*(p_energy - math.log(math.exp(-pow(img[0,x-1,y-1]-label[0],2)/pow(100,2))))
                unaries[1,x-1,y-1]=(n_energy- math.log(math.exp(-pow(img[0,x-1,y-1]-label[1],2)/pow(100,2))))
               # print(unaries[[0,1],x-1,y-1],img[0,x-1,y-1])
            elif(anno[k]==1):
                unaries[1,x-1,y-1]=(p_energy- math.log(math.exp(-pow(img[0,x-1,y-1]-label[1],2)/pow(100,2))))
                unaries[0,x-1,y-1]=10*(n_energy- math.log(math.exp(-pow(img[0,x-1,y-1]-label[0],2)/pow(100,2))))
                #print(unaries[[0,1],x,y],img[0,x-1,y-1])
            
    #unaries[1,:,:]=20-img[0,:,:]#1 for images in 0-1 range
    return unaries


"""   
def main():
    network=torch.rand(1,1,10,10)
    #img = mpimg.imread('toy_image.png')
    #img= mpimg.imread('../0ng/export/00ng_im3_NADH_PC102_60s_40x_oil_intensity_image.bmp');
    img=torch.rand(256,256,1)
    img = img[:,:,0]
    img = resize(img, (128, 128), anti_aliasing=True)
    n=img.shape
    m=n[1]
    n=n[0]

    label = np.loadtxt("label_00ng_im3_NADH_PC102_60s_40x.txt", dtype='i', delimiter=',')
   
    anno=torch.from_numpy(label)
    
    
    #img=np.random.randint(0,255,(50,50))
    opt=CRFasRNN(network, 10, 2 , 5, .1, .1, .1, None).cuda()# 2nd argument should be number of classes, set to 2
    image=torch.zeros(5,n,m).cuda();##not a easy fix for setting number of chanels to 1
    image[0,:,:]=torch.from_numpy(img)
    image[1,:,:]=torch.from_numpy(img)
    #image=torch.rand(5,128,128) # trying to merge with labels
    #image=image.cuda()
    #image=torch.rand(2,50,50)
    #print(image.type().is_cuda())
    y=opt(image,anno).cuda()
    y=y.squeeze(0)
    y1=np.zeros((n,m),dtype=np.double);
    y1=y[0,:,:];
    y2=y1.cpu().detach().numpy()
    
    np.savetxt('result.txt', y2, fmt='%.0e')
    plt.imshow(y2, interpolation='nearest')
    plt.show()
    
main()
"""
