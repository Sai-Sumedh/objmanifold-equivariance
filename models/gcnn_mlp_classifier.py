from torch import nn
from functions_geometry.utils import *
import torch
# from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
# from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling
from layers.gconvlayer import GCNN_layer

# class p4MaxPool(nn.Module):
#     def __init__(self, kernel, stride):
#         super().__init__()
#         self.kernel = kernel
#         self.stride = stride
#     def forward(self, x):
#         return plane_group_spatial_max_pooling(x, self.kernel, self.stride)
    
class GCNNMLPModel(nn.Module):
    """
    Defines a Classifier model
    Architecture:
    flatten image -> linear+relu -> local -> linear -> classify
    """
    def __init__(self, image_dim = 28, image_channels=1, kernel_size=3,
        out_dim = 10, batchnorm=True, bn_train=True, conv_bias=True, grouporder=4, numchan = 10):
        
        super().__init__()
        self.grouporder = grouporder
        self.theta = 360.0/self.grouporder #in degrees
        self.image_dim = image_dim        

        dim_conv1 = get_dim_after_conv(image_dim, kernel_size, 1)
        dim_conv2 = get_dim_after_conv(dim_conv1, kernel_size, 1)
        dim_conv3x = get_dim_after_conv(dim_conv2, 2, 2) #maxpool
        dim_conv3 = get_dim_after_conv(dim_conv3x, kernel_size, 1) #conv

        dim_conv4 = get_dim_after_conv(dim_conv3, kernel_size, 1) #conv

        # dim_conv5 = get_dim_after_conv(dim_conv4, kernel_size, 1) #conv
        # dim_conv6 = get_dim_after_conv(dim_conv5, kernel_size, 1) #conv
        # dim_conv7 = get_dim_after_conv(dim_conv6, 4, 1)
        
        
        numchan_withrot = numchan*self.grouporder
        if batchnorm:
            self.feature_extractor = nn.Sequential(
                GCNN_layer(image_channels, numchan,kernel_size, bias=conv_bias, theta=self.theta),
                nn.BatchNorm2d(numchan_withrot,affine=bn_train),
                nn.ReLU(),
                GCNN_layer(numchan_withrot, numchan,kernel_size, bias=conv_bias, theta=self.theta),
                nn.BatchNorm2d(numchan_withrot,affine=bn_train),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                GCNN_layer(numchan_withrot, numchan,kernel_size, bias=conv_bias, theta=self.theta),
                nn.BatchNorm2d(numchan_withrot,affine=bn_train),
                nn.ReLU(),
                GCNN_layer(numchan_withrot, numchan,kernel_size, bias=conv_bias, theta=self.theta),
                nn.BatchNorm2d(numchan_withrot,affine=bn_train),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(numchan_withrot*(dim_conv4**2), 20),
                nn.BatchNorm1d(20, affine=bn_train),
                nn.ReLU()
                # nn.MaxPool2d(2,2),
                # GCNN_layer(numchan_withrot, numchan,kernel_size, bias=conv_bias, theta=self.theta),
                # nn.BatchNorm2d(numchan_withrot,affine=bn_train),
                # nn.ReLU(),
                # GCNN_layer(numchan_withrot, numchan,kernel_size, bias=conv_bias, theta=self.theta),
                # nn.BatchNorm2d(numchan_withrot,affine=bn_train),
                # nn.ReLU(),
                # GCNN_layer(numchan_withrot, 10,4, bias=conv_bias, theta=self.theta),
             )
        else:
            self.feature_extractor = nn.Sequential(
                GCNN_layer(image_channels, numchan,kernel_size, bias=conv_bias, theta=self.theta),
                # nn.BatchNorm2d(numchan_withrot,affine=bn_train),
                nn.ReLU(),
                GCNN_layer(numchan_withrot, numchan,kernel_size, bias=conv_bias, theta=self.theta),
                # nn.BatchNorm2d(numchan_withrot,affine=bn_train),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                GCNN_layer(numchan_withrot, numchan,kernel_size, bias=conv_bias, theta=self.theta),
                # nn.BatchNorm2d(numchan_withrot,affine=bn_train),
                nn.ReLU(),
                GCNN_layer(numchan_withrot, numchan,kernel_size, bias=conv_bias, theta=self.theta),
                # nn.BatchNorm2d(numchan_withrot,affine=bn_train),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(numchan_withrot*(dim_conv4**2), 20),
                # nn.BatchNorm1d(20, affine=bn_train),
                nn.ReLU()
             )
        self.classifier = nn.Linear(20, out_dim)

    def forward(self, x):
        
        x = self.feature_extractor(x)
        xout = self.classifier(x)

        return xout