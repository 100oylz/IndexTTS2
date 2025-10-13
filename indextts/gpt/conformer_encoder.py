
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from indextts.gpt.conformer.attention import (MultiHeadedAttention,
                                              RelPositionMultiHeadedAttention)
from indextts.gpt.conformer.embedding import (NoPositionalEncoding,
                                              PositionalEncoding,
                                              RelPositionalEncoding)
from indextts.gpt.conformer.subsampling import (Conv2dSubsampling2,
                                                Conv2dSubsampling4,
                                                Conv2dSubsampling6,
                                                Conv2dSubsampling8,
                                                LinearNoSubsampling)
from indextts.utils.common import make_pad_mask

class FeedForwardModule(nn.Module):
    def __init__(self,input_dim,hidden_dim,dropout_rate,activation=nn.ReLU()):
        super().__init__()
        self.w_1 = nn.Linear(input_dim,hidden_dim)
        self.w_2 = nn.Linear(hidden_dim,input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation
    def forward(self,x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    
class ConvulationModule(nn.Module):
    def __init__(self,channels,kernel_size:int=15,activation:torch.nn.Module=nn.ReLU(),bias:bool=True):
        super().__init__()
        self.pointwise_conv1=nn.Conv1d(channels,2*channels,kernel_size=1,stride=1,padding=1,bias=bias)
        
        assert kernel_size%2==1,"kernel_size must be odd number"
        self.lorder = 0
        
        self.depthwise_conv=nn.Conv1d(channels,channels,kernel_size=kernel_size,stride=1,padding=(kernel_size-1)//2,groups=channels,bias=bias)
        
        self.pointwise_conv2=nn.Conv1d(channels,channels,kernel_size=1,stride=1,padding=0,bias=bias)
        
        self.activation=activation
        self.use_layer_norm=True
        self.norm = nn.LayerNorm(channels)
        
    def forward(self,x):
        
        x=x.transpose(1,2)
        if(self.use_layer_norm):
            x=self.norm(x.transpose(1,2)).transpose(1,2)
        x=self.pointwise_conv1(x)
        x=F.glu(x)