import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from Modules import *



class ScaledDotProductAttention_hyper(nn.Module):
    ''' Scaled Dot-Product Attention for Hypergraph'''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = attn_dropout

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn =  F.dropout(F.softmax(attn, dim=-1), self.dropout, training=self.training)
        output = torch.matmul(attn, v)

        return output, attn

class HyperGraphAttentionLayerSparse(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, transfer, concat=True, bias=False):
        super(HyperGraphAttentionLayerSparse, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat


        self.transfer = transfer

        if self.transfer:
            self.weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.register_parameter('weight', None)

        self.weight2 = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight3 = Parameter(torch.Tensor(self.out_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.word_context = nn.Embedding(1, self.out_features)
      
       
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.attention1 = ScaledDotProductAttention_hyper(temperature=self.out_features ** 0.5, attn_dropout = self.dropout)
        self.attention2 = ScaledDotProductAttention_hyper(temperature=self.out_features ** 0.5, attn_dropout = self.dropout)
        

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
        nn.init.uniform_(self.word_context.weight.data, -stdv, stdv)


    def forward(self, x, adj):
        residual = x

        x_4att = x.matmul(self.weight2)


        if self.transfer:
            x = x.matmul(self.weight)
            if self.bias is not None:
                x = x + self.bias        

        N1 = adj.shape[1] #number of edge
        N2 = adj.shape[2] #number of node


        q1 = self.word_context.weight[0:].view(1, 1, -1).repeat(x.shape[0], N1, 1).view(x.shape[0], N1, self.out_features)
        edge, att1 = self.attention1(q1, x_4att, x, mask = adj) 


        edge_4att = edge.matmul(self.weight3)
      
        node, attn = self.attention2(x_4att, edge_4att, edge, mask = adj.transpose(1, 2)) 


        if self.concat:

            node = F.relu(node)
            edge = F.relu(edge)

        return node, edge

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
