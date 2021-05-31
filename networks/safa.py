import torch.nn as nn
import torch.nn.functional as F
from networks import backbones
import torch

class SA(nn.Module):
    def __init__(self, in_dim, num=8):
        super().__init__()
        hid_dim = in_dim // 2
        self.w1, self.b1 = self.init_weights_(in_dim, hid_dim, num)
        self.w2, self.b2 = self.init_weights_(hid_dim, in_dim, num)

    def init_weights_(self, din, dout, dnum):
        weight = torch.empty(din, dout, dnum)
        nn.init.normal_(weight, mean=0.0, std=0.005)
        bias = torch.empty(1, dout, dnum)
        nn.init.constant_(bias, val=0.1)
        weight = torch.nn.Parameter(weight)
        bias = torch.nn.Parameter(bias)
        return weight, bias

    def forward(self, x):
        mask, _ = x.max(1)
        mask = torch.einsum('bi, ijd -> bjd', mask, self.w1) + self.b1
        mask = torch.einsum('bjd, jid -> bid', mask, self.w2) + self.b2
        return mask

class SAFA(nn.Module):

    def __init__(self, sa_num=8, H1=112, W1=616, H2=112, W2=616):
        super().__init__()

        self.extract2 = backbones.ResNet34()
        in_dim1 = (H1 // 8) * (W1 // 8)
        in_dim2 = (H2 // 8) * (W2 // 8)
        self.sa1 = SA(in_dim1, sa_num)
        self.sa2 = SA(in_dim2, sa_num)

    def forward(self, im2, res1):
        # Local feature extraction
        f2 = self.extract2(im2)
        B, C, _, _ = res1.shape
        f1 = res1.view(B, C, -1)
        f2 = f2.view(B, C, -1)

        # Spatial aware attention
        w1 = self.sa1(f1)
        w2 = self.sa2(f2)

        # Global feature aggregation
        f1 = torch.matmul(f1, w1).view(B, -1)
        f2 = torch.matmul(f2, w2).view(B, -1)

        # Feature reduction
        f1 = F.normalize(f1, p=2, dim=1)
        f2 = F.normalize(f2, p=2, dim=1)

        return f1, f2

