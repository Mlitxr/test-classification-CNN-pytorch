import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class cnnmodel(nn.Module):
    def __init__(self):
        super(cnnmodel,self).__init__()

        class_num=10#十分类
        Channel_in = 1
        Channel_out = kernel_num
        Kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList([nn.Conv2d(Channel_in, Channel_out,(K, word2vec_size)) for K in Kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Kernel_sizes)*Channel_out, class_num)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x,1)
        x = self.dropout(x)
        output = self.fc1(x)
        return output

if __name__ == '__main__':
    model = cnnmodel()
    input = torch.ones([1,128,128])
    print(model(input).size())