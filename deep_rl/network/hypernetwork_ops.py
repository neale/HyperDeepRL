import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_utils import *

def linear(x):
    return x


class LinearMixer(nn.Module):
    def __init__(self, config):
        super(Mixer, self).__init__()
        self.s = config.s_dim
        self.z = config.z_dim
        self.bias = config.bias
        self.n_gen = config.n_gen
        try:
            self.act = getattr(torch.nn.functional, config.act)
        except:
            self.act_out = linear
        try:
            self.act_out = getattr(torch.nn.functional, config.act_out)
        except:
            self.act_out = linear
        self.d_output = config.d_output
        self.act = getattr(torch.nn.functional, config.act)
        self.act_out = getattr(torch.nn.functional, config.act_out)
        self.d_hidden = config.d_hidden
        self.linear1 = layer_init(nn.Linear(self.s, self.d_hidden))#, bias=self.bias))
        self.linear2 = layer_init(nn.Linear(self.d_hidden, self.z*self.n_gen))#, bias=self.bias))

    def forward(self, x):
        x = self.act(self.linear1(x))
        x = self.act_out(self.linear2(x))
        x = x.view(-1, self.n_gen, self.z)
        w = torch.stack([x[:, i] for i in range(self.n_gen)])
        return w


class LinearGenerator(nn.Module):
    def __init__(self, config):
        super(LinearGenerator, self).__init__()
        self.z = config.z_dim
        self.bias = config.bias
        try:
            self.act = getattr(torch.nn.functional, config.act)
        except:
            self.act_out = linear
        try:
            self.act_out = getattr(torch.nn.functional, config.act_out)
        except:
            self.act_out = linear
        self.d_output = config.d_output
        self.d_input = config.d_input
        self.d_hidden = config.d_hidden
        
        self.linear1 = nn.Linear(self.z, self.d_hidden, bias=self.bias)
        self.linear2 = nn.Linear(self.d_hidden, self.d_output * self.d_input, bias=self.bias)
    
    def forward(self, z, x=None, theta=None):
        """ HyperModel Core
        inputs:
            z: Random sample used to generate weights [n_particles x noise_dim]
            x (optional): data to evaluate generated model on. If provided, weights are thrown away after use
            theta (optional): previously generated weights. if provided, no additional weights are generated
        outputs: 
            if x: then f(x ; g(z)) 
            if theta: then f(x ; theta)
            else: g(z)
        """
        if theta is None:
            z = self.act(self.linear1(z))
            z = self.act_out(self.linear2(z))
            w, b = z[:, :self.d_output*self.d_input], z[:, -self.d_output:]
            w = w.view(-1, self.d_output, self.d_input)
            b = b.view(-1, 1, self.d_output)
        else:
            w, b = theta
        if x is not None:
            x = torch.baddbmm(b, x, w.transpose(1,2)) # fused op is faster than list-comp over F.linear
            return x
        return w, b#.squeeze(1)

class ConvGenerator(nn.Module):
    def __init__(self, config):
        super(ConvGenerator, self).__init__()
        self.z = config.z_dim
        self.k = config.kernel
        self.bias = config.bias
        try:
            self.act = getattr(torch.nn.functional, config.act)
        except:
            self.act_out = linear
        try:
            self.act_out = getattr(torch.nn.functional, config.act_out)
        except:
            self.act_out = linear

        self.d_output = config.d_output
        self.d_input = config.d_input
        self.d_hidden = config.d_hidden
        
        self.linear1 = nn.Linear(self.z, self.d_hidden, bias=self.bias)
        self.linear2 = nn.Linear(self.d_hidden, self.d_output * self.d_input * self.k * self.k + self.d_output, bias=self.bias)
    
    def forward(self, z, x=None, stride=1, theta=None):
        if theta is None:
            z = self.act(self.linear1(z))
            z = self.act_out(self.linear2(z))
            w = z[:, :self.d_output*self.d_input*self.k*self.k]
            b = z[:, -self.d_output:]
            w = w.view(-1, self.d_output, self.d_input, self.k, self.k)
            b = b.view(-1, self.d_output)
        else:
            w, b = theta
        if x is not None:
            x = torch.stack([F.conv2d(x[i], w[i], bias=b[i], stride=stride) for i in range(w.shape[0])])
            return x
        return w, b.squeeze(1)

