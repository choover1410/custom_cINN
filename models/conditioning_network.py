import torch
import torch.nn as nn
from models.constants import *

class conditioning_network(nn.Module):
    '''conditioning network
        The input to the conditioning network are the observations (y)
        Args: 
        y: Observations (B X Obs)
    '''
    def __init__(self):
        super().__init__()

        class Flatten(nn.Module):
            def __init__(self, *args):
                super().__init__()
            def forward(self, x):
                return x.view(x.shape[0], -1)

        class Unflatten(nn.Module):
            def __init__(self, *args):
                super().__init__()
            def forward(self, x):
                if x[:,0,0].shape == (16,):
                    out = x.view(16,1,8,8) # for config_1  change this to out = x.view(16,2,8,8)
                elif x[:,0,0].shape == (1000,):
                    out = x.view(1000,1,8,8) # for config_1  change this to out = x.view(1000,2,8,8)
                elif x[:,0,0].shape == (1,):
                    out = x.view(1,1,8,8) # for config_1  change this to out = x.view(1,2,8,8)
                return out

        block_1_arb_chans = 21
        block_2_arb_chans = 96

        self.multiscale = nn.ModuleList([
                           # C3 (Block 3)
                           nn.Sequential(Unflatten(),
                                         nn.ConvTranspose2d(1, block_1_arb_chans, 2, padding=0), # for config_1  change this to nn.ConvTranspose2d(2,  48, 2, padding=0)
                                         nn.ReLU(inplace=True),
                                         nn.ConvTranspose2d(block_1_arb_chans, c3_size, 2, padding=1,stride=2)),
                           # C2 (Block 2)
                           nn.Sequential(nn.ReLU(inplace=True),
                                         nn.ConvTranspose2d(c3_size,  block_2_arb_chans, 2, padding=0,stride=2),
                                         nn.ReLU(inplace=True),
                                         nn.ConvTranspose2d(block_2_arb_chans, c2_size, 3, padding=1, stride=1)),
                           # C1 (Block 1)
                           nn.Sequential(nn.ReLU(inplace=True),
                                         nn.ConvTranspose2d(c2_size, c1_size, 2, padding=0, stride=2)),
                           # C4 (Block 4)
                           nn.Sequential(nn.ReLU(inplace=True),
                                         nn.AvgPool2d(6),
                                         Flatten(),
                                         nn.Linear(12000, 9600),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(9600, 6400),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(6400, 4800),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(4800, 2048),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(2048, 1024),                                         
                                         nn.ReLU(inplace=True),
                                         nn.Linear(1024, c4_size))])
                                         

    def forward(self, cond):
        val_cond = [cond]
        for val in self.multiscale:
            val_cond.append(val(val_cond[-1]))
        return val_cond[1:]
