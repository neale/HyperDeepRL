#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
from .hypernetwork_ops import *
from ..utils.hypernet_bodies_defs import *

particles = 2

class NatureConvHyperBody(nn.Module):
    def __init__(self, in_channels=4):
        super(NatureConvHyperBody, self).__init__()
        self.mixer = False
        self.feature_dim = 512
        conf = NatureConvBody_config(in_channels, self.feature_dim)
        self.n_gen = conf['n_gen']
        self.z_dim = conf['z_dim']

        self.conv1 = ConvGenerator(conf['conv1'])
        self.conv2 = ConvGenerator(conf['conv2'])
        self.conv3 = ConvGenerator(conf['conv3'])
        self.fc4 = LinearGenerator(conf['fc4'])

    def forward(self, x):
        if not self.mixer:
            z = torch.rand(self.n_gen, particles, self.z_dim)
        ph = [1] * len(x.shape)
        x = x.unsqueeze(0).repeat(particles, *ph)
        y = F.relu(self.conv1(z[0], x, stride=4))
        y = F.relu(self.conv2(z[1], y, stride=2))
        y = F.relu(self.conv3(z[2], y, stride=1))
        y = y.view(particles, y.size(1), -1)
        y = F.relu(self.fc4(z[3], y))
        print ('body return', x.shape)
        return y


class DDPGConvHyperBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DDPGConvHyperBody, self).__init__()
        self.mixer = False
        self.feature_dim = 39 * 39 * 32
        conf = DDPGConvBody_config(in_channels, feature_dim)
        self.n_gen = conf['n_gen']
        self.z_dim = conf['z_dim']
        self.conv1 = ConvGenerator(conf['conv1'])
        self.conv2 = ConvGenerator(conf['conv2'])

    def forward(self, x):
        if not self.mixer:
            z = torch.rand(self.n_gen, particles, self.z_dim)
        x = x.unsqueeze(0).repeat(particles, 1, 1)
        y = F.elu(self.conv1(z[0], x, stride=2))
        y = F.elu(self.conv2(z[1], y, stride=1))
        y = y.view(y.size(0), -1)
        return y


class FCHyperBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCHyperBody, self).__init__()
        self.mixer = False
        dims = (state_dim,) + hidden_units
        conf = FCBody_config(state_dim, hidden_units, gate)
        self.n_gen = conf['n_gen']
        self.z_dim = conf['z_dim']
        self.gate = gate
        self.feature_dim = dims[-1]
        n_layers = conf['n_gen']
        self.layers = nn.ModuleList([LinearGenerator(conf['fc{}'.format(i+1)]) for i in range(n_layers)])

    def forward(self, x):
        if not self.mixer:
            z = torch.rand(self.n_gen, particles, self.z_dim)
        x = x.unsqueeze(0).repeat(particles, 1, 1)
        for i, layer in enumerate(self.layers):
            x = self.gate(layer(z[i], x))
        print ('body return', x.shape)
        return x


class TwoLayerFCHyperBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), gate=F.relu):
        super(TwoLayerFCHyperBodyWithAction, self).__init__()
        self.mixer = False
        hidden_size1, hidden_size2 = hidden_units
        conf = TwoLayerFCBodyWithAction_config(state_dim, action_dim, hidden_units, gate)
        self.n_gen = conf['n_gen']
        self.z_dim = conf['z_dim']
        self.fc1 = LinearGenerator(conf['fc1'])
        self.fc2 = LinearGenerator(conf['fc2'])
        self.gate = gate
        self.feature_dim = hidden_size2

    def forward(self, x, action):
        if not self.mixer:
            z = torch.rand(self.n_gen, particles, self.z_dim)
        x = x.unsqueeze(0).repeat(particles, 1, 1)
        x = self.gate(self.fc1(z[0], x))
        if x.shape[0] != action.shape[0]:
            action = action.unsqueeze(0).repeat(particles, 1, 1)
        phi = self.gate(self.fc2(z[1], torch.cat([x, action], dim=2)))
        return phi


class OneLayerFCHyperBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, gate=F.relu):
        super(OneLayerFCHyperBodyWithAction, self).__init__()
        self.mixer = False
        conf = OneLayerFCBodyWithAction_config(state_dim, action_dim, hidden_units, gate, self.mixer)
        self.n_gen = conf['n_gen']
        self.z_dim = conf['z_dim']
        self.fc_s = LinearGenerator(conf['fc_s'])
        self.fc_a = LinearGenerator(conf['fc_a'])
        self.gate = gate
        self.feature_dim = hidden_units * 2

    def forward(self, x, action):
        if not self.mixer:
            z = torch.rand(self.n_gen, particles, self.z_dim)
        x = x.unsqueeze(0).repeat(particles, 1, 1)
        phi = self.gate(torch.cat([self.fc_s(z[0], x), self.fc_a(z[1], action)], dim=1))
        return phi


class DummyHyperBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyHyperBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x
