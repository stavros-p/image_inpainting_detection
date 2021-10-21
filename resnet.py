# -*- coding: utf-8 -*-
"""
Custom Resnet implementation according to the paper
Localization of deep inpainting using HP - FCN 
paper: https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Localization_of_Deep_Inpainting_Using_High-Pass_Fully_Convolutional_Network_ICCV_2019_paper.pdf

Code Implementation: Stavros Papadopoulos
May 2021
"""

import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F

class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(block, self).__init__()
        self.expansion = 4 # output channels/input channel
        self.bn1 = nn.BatchNorm2d(in_channels)
        # 1x1 convolution 
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        # 3x3 convolution
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels)
        # 1x1 convolution
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=stride, #the last conv on the 2nd bottleneck should have stride of  2
            padding=0,
            bias=False
        )
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride


    def forward(self, x):
        # we'll use this block multiple times later on
        identity = x.clone()
        # BN/RELU + 1x1, stride=1
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        
        # BN/RELU + 3x3, stride=1        
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        # BN/RELU + 1x1, stride=1 or 2
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module): 
    '''
    # layer: how many times we want to use the block (how many bottlenecks to have)
    # Image channels : In our case 3 
    # num_classes = 2
    '''
    def __init__(self, block):
        super(ResNet, self).__init__()
        self.in_channels = 32
        
        # pf convolution 3-D -> 9-D
        self.pf_conv = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1, bias=False)

        # conv1 maps m x n x 9 to m x n x 32
        self.conv1 = nn.Conv2d(9, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, intermediate_channels=32, stride=2
        )
        self.layer2 = self._make_layer(
            block, intermediate_channels=64, stride=2
        )
        self.layer3 = self._make_layer(
            block, intermediate_channels=128, stride=2
        )
        self.layer4 = self._make_layer(
            block, intermediate_channels=256, stride=2
        )
        self.upsample1 = nn.ConvTranspose2d(1024, 64, 8, stride=4, padding=2)
        self.upsample2 = nn.ConvTranspose2d(64, 4, 8, stride=4, padding=2)
        self.weak_conv = nn.Conv2d(4, 1, 5, stride=1, padding=2)
        self.sigm = nn.Sigmoid()

        # gets the initial pf kernels 
        self.pf_list = get_pf_list()
        
        #resets the initial pf kernels to be equal to pflist
        self.reset_pf()

    def forward(self, x):
        
        h_init,w_init= x.shape[2], x.shape[3]
        # pass through hp - module
        x = self.pf_conv(x)
       
        # maps 9-d to 32-d
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # build the 4 resnet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # upsampling in 2 stage
        x = self.upsample1(x)
        x = self.upsample2(x)

        # eliminate checkerboard artifactis of transpose (upsampling) convolutions
        x = self.weak_conv(x)
        x = F.interpolate(x, size=(h_init,w_init), mode='nearest')
        # Logits are fed to a sigmoid layer for classification
        # yielding the localization map with pixel - wise predictions.
        x = self.sigm(x)

        return x

    def reset_pf(self):
        for idx, pf in enumerate(self.pf_list):
            self.pf_conv.weight.data[idx, :, :, :] = pf

    def _make_layer(self, block, intermediate_channels, stride):
        # Creates the layer ( = Block with 2 bottlenecks)
        # num_residual_blocks = number of time is going to use the block (=2)
        # intermediate_channels = out_channels/4
        # Stride: one of the residual block will have a stride of 2
        identity_downsample = None 
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        # defines tges identity_downsample when the dimensions of x have to change

        depth_upsample = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(
                self.in_channels,
                intermediate_channels * 4,
                kernel_size=1,
                stride=1,
                bias=False
            )
        )

        #this is the layer that changes the #channels
        # the first bottleneck of the block
        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample = depth_upsample, stride=1)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        dim_downsample = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(
                self.in_channels,
                intermediate_channels * 4,
                kernel_size=1,
                stride=stride,
                bias=False
            )
        )

        layers.append(block(self.in_channels, intermediate_channels, identity_downsample=dim_downsample, stride=2))

        return nn.Sequential(*layers)

# The initial hp kernels 
def get_pf_list():
    pf1 = np.array([[0, 0, 0],
                    [0, -1, 0],
                    [0, 1, 0]]).astype('float32')

    pf2 = np.array([[0, 0, 0],
                    [0, -1, 1],
                    [0, 0, 0]]).astype('float32')

    pf3 = np.array([[0, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]]).astype('float32')

    return [torch.tensor(pf1).clone(),
            torch.tensor(pf2).clone(),
            torch.tensor(pf3).clone(),
            torch.tensor(pf1).clone(),
            torch.tensor(pf2).clone(),
            torch.tensor(pf3).clone(),
            torch.tensor(pf1).clone(),
            torch.tensor(pf2).clone(),
            torch.tensor(pf3).clone()
            ]