import os
import copy

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__(in_features, out_features)
        self.epsilon = 1e-9

    def relprop(self, R):
        V = torch.clamp(self.weight, min=0)
        Z = torch.mm(self.X, torch.transpose(V, 0, 1)) + self.epsilon

        S = torch.div(R, Z)
        C = torch.mm(S, V)
        return self.X * C


class BatchNorm2d(nn.BatchNorm2d):
    def relprop(self, R):
        S = self.weight*torch.pow(self.running_var + self.eps, -0.5)
        S = torch.stack([torch.zeros(self.X.shape[-2], self.X.shape[-1]) + mean for mean in S])
        S = S.type(torch.cuda.FloatTensor)

        return torch.div(self.X * S * R, self.Y)


class ReLU(nn.ReLU):
    def relprop(self, R):
        return R


class MaxPool2d(nn.MaxPool2d):
    def gradprop(self, DY):
        temp, indices = F.max_pool2d(self.X,
                                     self.kernel_size,
                                     self.stride,
                                     self.padding,
                                     self.dilation,
                                     self.ceil_mode,
                                     True)
        DX = F.max_unpool2d(DY,
                            indices,
                            self.kernel_size,
                            self.stride,
                            self.padding)
        return DX

    def relprop(self, R):
        Z = self.Y + 1e-9
        S = torch.div(R, Z)
        C = self.gradprop(S)
        return self.X * C


class Conv2d(nn.Conv2d):
    def conv2d_forward(self, X, W, b):
        out = F.conv2d(X,
                       W,
                       b,
                       self.stride,
                       self.padding)
        return out

    def gradprop(self, DY, W):
        output_padding = self.X.size()[2] - ((self.Y.size()[2] - 1) * self.stride[0] - 2 * self.padding[0]
                                             + self.kernel_size[0])
        return F.conv_transpose2d(DY,
                                  W,
                                  stride=self.stride,
                                  padding=self.padding,
                                  output_padding=output_padding)

    def relprop(self, R):
        zero_bias = copy.deepcopy(self.bias)
        zero_bias = zero_bias * 0

        w_pos = torch.clamp(self.weight, min=0)

        Z = self.conv2d_forward(self.X, w_pos, zero_bias) + 1e-9
        S = R / Z
        C = self.gradprop(S, w_pos)
        return self.X * C


class Conv2dZbeta(nn.Conv2d):
    def gradprop(self, DY, W):
        output_padding = self.X.size()[2] - ((self.Y.size()[2] - 1) * self.stride[0] - 2 * self.padding[0]
                                             + self.kernel_size[0])
        return F.conv_transpose2d(DY, W, stride=self.stride,
                                  padding=self.padding, output_padding=output_padding)

    def conv2d_forward(self, X, W, b):
        out = F.conv2d(X, W, b, self.stride, self.padding)
        return out

    def relprop(self, R):
        nweight = torch.clamp(self.weight, max=0)
        pweight = torch.clamp(self.weight, min=0)

        L = self.X * 0 + torch.min(self.X)
        H = self.X * 0 + torch.max(self.X)

        zero_bias = copy.deepcopy(self.bias)

        zero_bias = zero_bias * 0

        Z = self.conv2d_forward(self.X, self.weight, zero_bias) \
            - self.conv2d_forward(L, pweight, zero_bias) \
            - self.conv2d_forward(H, nweight, zero_bias) + 1e-9

        S = R / Z

        R = self.X * self.gradprop(S, self.weight) - L * self.gradprop(S, pweight) - H * self.gradprop(S, nweight)
        return R


