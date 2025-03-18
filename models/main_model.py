import numpy as np 
import torch
import torch.nn as nn
from models.CouplingBlock import CouplingBlock
from models.CouplingOneSide import CouplingOneSide
from models.Downsample_model import Downsample
from models.constants import *

class Divide_data(nn.Module):
    '''Args:
        X: input (BXD) to  output (BXCXHXW) 
        (This is used to split the data  for the concat part for Z and 
        the other part for the network during the training and sampling phase).
    '''
    def __init__(self, input_dimension, split_data_channel):
        super(Divide_data,self).__init__()
        self.split_data_channel = split_data_channel
    def forward(self, x, sample_the_data=False):
        out = torch.split(x, self.split_data_channel,1)
        return out

class Flat_data(nn.Module):
    '''Args:
        X: input (BXCXHXW) 
        y: output (BXD)
        (This is used to flatten the data from 4D to 2D for the concat part of the fully connected layer).
    '''
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        return y

class Permute_data(nn.Module):
    '''
    Args: 
    x: input (BXCXHXW)
    To permute the data channel-wise. This operation called during both the training and testing.
    '''

    def __init__(self, input_data, seed):
        super(Permute_data,self).__init__()
        np.random.seed(seed)
        self.Permute_data = np.random.permutation(input_data)
        np.random.seed()
        Permute_sample = np.zeros((self.Permute_data.shape))
        for i, j in enumerate(self.Permute_data):
            Permute_sample[j] = i
        self.Permute_sample = Permute_sample

    def forward(self, x, sample_the_data=False):
        if sample_the_data == False:
            y = x[:, self.Permute_data]
            return y
        else:
            y1 = x[:, self.Permute_sample]
            return y1

class Unflat_data(nn.Module):
    '''Args:
        X: input (BXD) to  output (BXCXHXW) 
        This is used to unflatten the data from 2D to 4D for the concat part during the sampling phase
    '''
    def __init__(self, input_dimension):
        super().__init__()
        self.shape_dim = input_dimension[0]

    def forward(self, x, sample_the_data=False):
        y = x.view(x.shape[0], *self.shape_dim)
        return y

class main_file(nn.Module):
    def __init__(self,
                s_net_t_net,
                s_net_t_net2,
                s_net_t_net3,
                ):

        super(main_file,self).__init__()
        # Misc.
        self.downsample = Downsample()
        self.flat2 = Flat_data()
        self.unflat1 = Unflat_data([(2,32,32)]) # for inverse
        self.unflat2 = Unflat_data([(4,16,16)]) # for inverse

        # Block 1
        # Last Output Channel Size of conditional Block 1 must match c_size
        self.single_side1 = CouplingOneSide(s_net_t_net, c1_size)
        self.single_side2 = CouplingOneSide(s_net_t_net, c1_size)
        self.single_side3 = CouplingOneSide(s_net_t_net, c1_size)
        self.single_side4 = CouplingOneSide(s_net_t_net, c1_size)
        self.single_side5 = CouplingOneSide(s_net_t_net, c1_size)
        self.single_side6 = CouplingOneSide(s_net_t_net, c1_size)
        # End Block 1

        # Block 2
        block2_input_dim = 4
        block2_length=3

        permute_a1=block2_input_dim
        self.coupling1 = CouplingBlock(s_net_t_net, block2_input_dim, block2_length, c2_size)
        self.coupling2 = CouplingBlock(s_net_t_net, block2_input_dim, block2_length, c2_size)
        self.coupling3 = CouplingBlock(s_net_t_net, block2_input_dim, block2_length, c2_size)
        self.coupling4 = CouplingBlock(s_net_t_net, block2_input_dim, block2_length, c2_size)
        self.coupling5 = CouplingBlock(s_net_t_net, block2_input_dim, block2_length, c2_size)
        self.permute = Permute_data(permute_a1, 0)
        self.permute_c1 = Permute_data(permute_a1, 1)
        self.permute_c2 = Permute_data(permute_a1, 2)
        self.permute_c3 = Permute_data(permute_a1, 3)
        self.permute_c4 = Permute_data(permute_a1, 4)
        self.split = Divide_data(block2_input_dim, [2,2])
        # End Block 2

        # Block 3
        block3_input_dim=8
        block3_length=3

        permute_a2=block3_input_dim
        self.coupling21 = CouplingBlock(s_net_t_net2, block3_input_dim, block3_length, c3_size)
        self.coupling22 = CouplingBlock(s_net_t_net2, block3_input_dim, block3_length, c3_size)
        self.coupling23 = CouplingBlock(s_net_t_net2, block3_input_dim, block3_length, c3_size)
        self.coupling24 = CouplingBlock(s_net_t_net2, block3_input_dim, block3_length, c3_size)
        self.permute2 = Permute_data(permute_a2,0)
        self.permute2_c1 = Permute_data(permute_a2,1)
        self.permute2_c2 = Permute_data(permute_a2,2)
        self.permute2_c3 = Permute_data(permute_a2,3)
        self.split2 = Divide_data(block3_input_dim, [4,4])
        # End Block 3

        # Block 4
        block4_input_dim=1024 # no more than 1024
        block4_length=1

        permute_a3=block4_input_dim
        self.coupling31 = CouplingBlock(s_net_t_net3, block4_input_dim, block4_length, c4_size)
        self.permute3 = Permute_data(permute_a3,0)
        # End Block 4


    def forward(self, x, c1,c2,c3,c4,sample_the_data=False,forward=False,jac=False):
        if forward==True:
            out1= self.single_side1(x,c1)
            jac0 = self.single_side1.jacobian()
            out2 = self.single_side2(out1,c1)
            jac0_1 = self.single_side2.jacobian()
            out3= self.single_side3(out2,c1)
            jac0_2 = self.single_side3.jacobian()
            out4 = self.single_side4(out3,c1)
            jac0_3 = self.single_side4.jacobian()
            out5 = self.single_side5(out4,c1)
            jac0_4 = self.single_side5.jacobian()
            out6 = self.single_side6(out5,c1)
            jac0_5 = self.single_side6.jacobian()
            out7 = self.downsample(out6)

            #2
            out12 = self.coupling1(out7,c2)
            jac1 = self.coupling1.jacobian()
            out13 = self.permute(out12)
            out14 = self.coupling2(out13,c2)
            jac1_c1 = self.coupling2.jacobian()
            out15 = self.permute_c1(out14)
            out16 = self.coupling3(out15,c2)
            jac1_c2 = self.coupling3.jacobian()
            out17 = self.permute_c2(out16)
            out18 = self.coupling4(out17,c2)
            jac1_c3 = self.coupling4.jacobian()
            out19 = self.permute_c3(out18)
            out20 = self.coupling5(out19,c2)
            jac1_c4 = self.coupling5.jacobian()
            out21 = self.permute_c4(out20)
            out22 = self.split(out21)
            out1s = out22[0] 
            out2s = out22[1] 

            flat_output1 = self.flat2(out2s)

            out31 = self.downsample(out1s)

            #3
            out32 = self.coupling21(out31,c3)
            jac2 = self.coupling21.jacobian()
            out33 = self.permute2(out32)
            out34 = self.coupling22(out33,c3)
            jac2_c1 = self.coupling22.jacobian()
            out35 = self.permute2_c1(out34)
            out36 = self.coupling23(out35,c3)
            jac2_c2 = self.coupling23.jacobian()
            out37= self.permute2_c2(out36)
            out38 = self.coupling24(out37,c3)
            jac2_c3 = self.coupling24.jacobian()
            out39 = self.permute2_c3(out38)
            out40 = self.split2(out39)
            out1s4 = out40[0] 
            out2s4 = out40[1] 
            flat_output2 = self.flat2(out2s4)        
            flat_ds2 = self.flat2(out1s4)  

            #4
            out1f = self.coupling31(flat_ds2,c4)
            jac3 = self.coupling31.jacobian()
            out_all = self.permute3(out1f)

            final_out  = torch.cat((flat_output1,flat_output2,out_all),dim=1)

            #jacobian
            jac = jac0+jac1+jac2+jac3+jac0_1+jac0_2+jac0_3+jac0_4+jac0_5+jac1_c1+jac1_c2+jac1_c3+jac1_c4+jac2_c1+jac2_c2+jac2_c3
            return final_out, jac
        else:

            #unflat the 2X32X32 data
            out1 = x[:,:2048]
            out1_unflat = self.unflat1(out1)
            #unflat the 4X16X16 data            
            out2 = x[:,2048:3072]
            out2_unflat = self.unflat2(out2)
            # this is considered as the  input to the INN model 1024 
            out3 = x[:,3072:]
            #permute the data
            out3p = self.permute3(out3,sample_the_data=True)
            # consider the INN model FC
            out = self.coupling31(out3p,c4,sample_the_data=True)
            out3_unflat = self.unflat2(out)
            #combine the data
            combine_out2_out3 = torch.cat((out3_unflat,out2_unflat), dim=1)
            #=========================================
            #permute the data
            out_4 =  self.permute2_c3(combine_out2_out3,sample_the_data=True)

            
            out_5 = self.coupling24(out_4,c3,sample_the_data=True)    
            #==============================================    
            #=========================================
            #permute the data
            out_4 =  self.permute2_c2(out_5,sample_the_data=True)

            
            out_5 = self.coupling23(out_4,c3,sample_the_data=True)   
            #==============================================   
            #=========================================
            #permute the data
            out_4 =  self.permute2_c1(out_5,sample_the_data=True)

            
            out_5 = self.coupling22(out_4,c3,sample_the_data=True)   
            #==============================================  Here 
            #=========================================
            #permute the data
            out_4 =  self.permute2(out_5,sample_the_data=True)

            
            out_5 = self.coupling21(out_4,c3,sample_the_data=True)   
            #==============================================  Here 

            #updample to 2X32X32
            out_6 = self.downsample(out_5,sample_the_data=True)
            #combine the data with out_1 4X32X32
            combine_out6_out1 = torch.cat((out_6,out1_unflat), dim=1)
            #=============================
            #permute
            out_7 =  self.permute_c4(combine_out6_out1,sample_the_data=True)
            
            out_8 = self.coupling5(out_7,c2,sample_the_data=True) 
            #==================================
            #=============================
            #permute
            out_7 =  self.permute_c3(out_8,sample_the_data=True)
 
            out_8 = self.coupling4(out_7,c2,sample_the_data=True) 
            #==================================
            #=============================
            #permute
            out_7 =  self.permute_c2(out_8,sample_the_data=True)
    
            out_8 = self.coupling3(out_7,c2,sample_the_data=True) 
 
            #==================================
            #=============================
            #permute
            out_7 =  self.permute_c1(out_8,sample_the_data=True)
            
            out_8 = self.coupling2(out_7,c2,sample_the_data=True)  
            #==================================
            #=============================
            #permute
            out_7 =  self.permute(out_8,sample_the_data=True)
            
            out_8 = self.coupling1(out_7,c2,sample_the_data=True)  
            #==================================
            #upsample 1X64X64
            out_9 = self.downsample(out_8,sample_the_data=True)
            out_10 = self.single_side6(out_9,c1,sample_the_data=True)
            out_10 = self.single_side5(out_10,c1,sample_the_data=True)
            out_10 = self.single_side4(out_10,c1,sample_the_data=True)
            out_10 = self.single_side3(out_10,c1,sample_the_data=True)
            out_10 = self.single_side2(out_10,c1,sample_the_data=True)
            out_10 = self.single_side1(out_10,c1,sample_the_data=True)

            return out_10
