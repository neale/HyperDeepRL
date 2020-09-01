#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
from .hypernetwork_ops import *
from ..utils.hypernet_bodies_defs import *
import numpy as np

class NatureConvHyperBody(nn.Module):
    def __init__(self, in_channels=4):
        super(NatureConvHyperBody, self).__init__()
        self.mixer = False
        self.feature_dim = 512
        self.config = NatureConvBody_config(in_channels, self.feature_dim)
        self.conv1 = ConvGenerator(self.config['conv1']).cuda()
        self.conv2 = ConvGenerator(self.config['conv2']).cuda()
        self.conv3 = ConvGenerator(self.config['conv3']).cuda()
        self.fc4 = LinearGenerator(self.config['fc4']).cuda()

    def forward(self, x=None, z=None, theta=None):
        # incoming x is batch of frames  [n, 4, width, height]
        x = x.unsqueeze(0).repeat(z.shape[1], 1, 1, 1, 1)    
        y = F.relu(self.conv1(z[0], x, stride=4))
        y = F.relu(self.conv2(z[1], y, stride=4))
        y = F.relu(self.conv3(z[2], y, stride=1))
        y = y.view(y.size(0), y.size(1), -1)
        y = F.relu(self.fc4(z[3], y))
        return y


class DDPGConvHyperBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DDPGConvHyperBody, self).__init__()
        self.mixer = False
        self.feature_dim = 39 * 39 * 32
        self.config = DDPGConvBody_config(in_channels, feature_dim)
        self.conv1 = ConvGenerator(self.config['conv1'])
        self.conv2 = ConvGenerator(self.config['conv2'])

    def forward(self, x=None, z=None):
        x = x.unsqueeze(0).repeat(particles, 1, 1)
        y = F.elu(self.conv1(z[0], x, stride=2))
        y = F.elu(self.conv2(z[1], y, stride=1))
        y = y.view(y.size(0), -1)
        return y

class ToyFCHyperBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(ToyFCHyperBody, self).__init__()
        self.mixer = False
        dims = (state_dim,) + hidden_units
        self.config = ToyFCBody_config(state_dim, hidden_units, gate)
        self.gate = gate
        self.feature_dim = dims[-1]
        n_layers = self.config['n_gen']
        self.layers = nn.ModuleList([LinearGenerator(self.config['fc{}'.format(i+1)]).cuda() for i in range(n_layers)])

    def forward(self, x=None, z=None, theta=None):
        if x is None:
            weights = []
            for i, layer in enumerate(self.layers):
                w, b = layer(z[i])
                weights.append(w)
                weights.append(b)
            return weights
        ones_mask = torch.ones(x.dim()).long().tolist()
        x = x.unsqueeze(0).repeat(z.shape[1], *ones_mask)
        if x.size(2) == 1:  # DM lab has incompatible sizing with gym
            x = x.squeeze(2)
        for i, layer in enumerate(self.layers):
            if theta:
                x = self.gate(layer(z[i], x, theta[i*2:(i*2)+2]))
            else:
                x = self.gate(layer(z[i], x))
        return x

class CartFCHyperBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(256, 256), gate=F.relu, hidden=None):
        super(CartFCHyperBody, self).__init__()
        self.mixer = False
        if hidden:
            hidden_units=(hidden, hidden)
        print (hidden_units)
        dims = (state_dim,) + hidden_units
        self.config = FCBody_config(state_dim, hidden_units, gate)
        self.gate = gate
        self.feature_dim = dims[-1]
        n_layers = self.config['n_gen']
        self.layers = nn.ModuleList([LinearGenerator(self.config['fc{}'.format(i+1)]).cuda() for i in range(n_layers)])

    def forward(self, x=None, z=None, theta=None):
        if x is None:
            weights = []
            for i, layer in enumerate(self.layers):
                w, b = layer(z[i])
                weights.append(w)
                weights.append(b)
            return weights
        ones_mask = torch.ones(x.dim()).long().tolist()
        x = x.unsqueeze(0).repeat(z.shape[1], *ones_mask)
        
        if x.size(2) == 1:  # DM lab has incompatible sizing with gym
            x = x.squeeze(2)
        
        for i, layer in enumerate(self.layers):
            x = self.gate(layer(z[i], x))
        return x


class FCHyperBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCHyperBody, self).__init__()
        self.mixer = False
        dims = (state_dim,) + hidden_units
        self.config = FCBody_config(state_dim, hidden_units, gate)
        self.gate = gate
        self.feature_dim = dims[-1]
        n_layers = self.config['n_gen']
        self.layers = nn.ModuleList(
                [LinearGenerator(self.config['fc{}'.format(i+1)]).cuda() for i in range(n_layers)])

    def generate_theta(self, seed):
        params = []
        for i, layer in enumerate(self.layers):
            params.append(layer(seed[i]))
        theta = torch.cat(params, -1)
        self.theta = theta
    
    def get_theta(self, layer):
        l1 = self.layers[0].d_output * self.layers[0].d_input + self.layers[0].d_output
        l2 = self.layers[1].d_output * self.layers[1].d_input + self.layers[1].d_output

        self.body_params_len = l1 + l2
        if layer == 'body1':
            theta = self.theta[:, :l1]
        elif layer == 'body2':
            theta = self.theta[:, l1:]
        return theta

    def forward(self, x=None, z=None, theta=False):
        if x is None:
            weights = []
            for i, layer in enumerate(self.layers):
                w, b = layer(z[i])
                weights.append(w)
                weights.append(b)
            return weights
        # x = x.unsqueeze(0).repeat(z.shape[1], 1, 1)
        ones_mask = torch.ones(x.dim()).long().tolist()
        x = x.unsqueeze(0).repeat(z.shape[1], *ones_mask)
        if x.size(2) == 1:  # DM lab has incompatible sizing with gym
            x = x.squeeze(2)
        for i, layer in enumerate(self.layers):
            if theta:
                params = self.get_theta('body{}'.format(i+1))
                x = self.gate(layer.evaluate(x, params))
            else:
                
                x = self.gate(layer.evaluate(x, layer(z[i])))
        return x


class FCHyperBodyFx(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCHyperBodyFx, self).__init__()
        self.mixer = False
        dims = (state_dim,) + hidden_units
        self.config = FCBody_config(state_dim, hidden_units, gate)
        self.gate = gate
        self.feature_dim = dims[-1]
        n_layers = self.config['n_gen']
        self.layers = nn.ModuleList([
            LinearGeneratorFx(
                self.config['fc{}'.format(i+1)]).cuda() for i in range(n_layers)])

    def forward(self, x=None, z=None, theta=None):
        if x is None:
            weights = []
            for i, layer in enumerate(self.layers):
                w, b = layer(z[i])
                weights.append(w)
                weights.append(b)
            return weights
        # x = x.unsqueeze(0).repeat(z.shape[1], 1, 1)
        ones_mask = torch.ones(x.dim()).long().tolist()
        x = x.unsqueeze(0).repeat(z.shape[1], *ones_mask)
        if x.size(2) == 1:  # DM lab has incompatible sizing with gym
            x = x.squeeze(2)
        for i, layer in enumerate(self.layers):
            if theta:
                x = self.gate(layer(z[i], x, theta[i*2:(i*2)+2]))
            else:
                x = self.gate(layer(z[i], x))
        return x


class DummyHyperBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyHyperBody, self).__init__()
        self.feature_dim = state_dim
        self.config = Dummy_config()

    def forward(self, x=None, z=None):
        if x is None:
            return []
        return x
