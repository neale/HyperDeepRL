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
        self.layers = nn.ModuleList([LinearGenerator(self.config['fc{}'.format(i+1)]).cuda() for i in range(n_layers)])

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


class TwoLayerFCHyperBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), gate=F.relu):
        super(TwoLayerFCHyperBodyWithAction, self).__init__()
        self.mixer = False
        hidden_size1, hidden_size2 = hidden_units
        self.config = TwoLayerFCBodyWithAction_config(state_dim, action_dim, hidden_units, gate)
        self.fc1 = LinearGenerator(self.config['fc1']).cuda()
        self.fc2 = LinearGenerator(self.config['fc2']).cuda()
        self.gate = gate
        self.feature_dim = hidden_size2

    def forward(self, x=None, action=None, z=None):
        if x is None:
            w1, b1 = self.fc1(z[0])
            w2, b2 = self.fc2(z[1])
            return [w1, b1, w2, b2]
        x = x.unsqueeze(0).repeat(particles, 1, 1)
        x = self.gate(self.fc1(z[0], x))
        if x.dim() != action.dim():
            action = action.unsqueeze(0).repeat(particles, 1, 1)
        phi = self.gate(self.fc2(z[1], torch.cat([x, action], dim=2)))
        return phi


class OneLayerFCHyperBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, gate=F.relu):
        super(OneLayerFCHyperBodyWithAction, self).__init__()
        self.mixer = False
        self.config = OneLayerFCBodyWithAction_config(state_dim, action_dim, hidden_units, gate, self.mixer)
        self.fc_s = LinearGenerator(self.config['fc_s'])
        self.fc_a = LinearGenerator(self.config['fc_a'])
        self.gate = gate
        self.feature_dim = hidden_units * 2

    def forward(self, x=None, action=None, z=None):
        if x is None:
            return (self.fc_s(z[0]), self.fc_a(z[1]))
        x = x.unsqueeze(0).repeat(particles, 1, 1)
        phi = self.gate(torch.cat([self.fc_s(z[0], x), self.fc_a(z[1], action)], dim=1))
        return phi


class DummyHyperBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyHyperBody, self).__init__()
        self.feature_dim = state_dim
        self.config = Dummy_config()

    def forward(self, x=None, z=None):
        if x is None:
            return []
        return x
