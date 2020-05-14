from collections import namedtuple

## If using python 3.7+, use this template for each nametuple. 
## The old way should still work though
# gen_config = namedtuple('layer', fields)
# defaults = (100, 3, False, 64, 64, 'relu', None, 6) # python 3.7

fields = ('d_output', 'd_input', 'd_hidden', 'kernel', 
          'bias', 's_dim', 'z_dim', 'act', 'act_out', 'n_gen')
defaults = (64, 3, False, 64, 64, 'relu', None, 4) # python 3.6

nature_conv_body = namedtuple('nature_conv_body', ' '.join(fields))
nature_conv_body.__new__.__defaults__ = defaults # python 3.6
def NatureConvBody_config(in_channels, feature_dim, mixer=False):
    net_config = {
            'conv1': nature_conv_body(d_input=in_channels, d_output=32, kernel=8),
            'conv2': nature_conv_body(d_input=32, d_output=64, kernel=4),
            'conv3': nature_conv_body(d_input=64, d_output=64, kernel=3),
            'fc4': nature_conv_body(d_input=576, d_output=feature_dim),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 4
            }
    if mixer:
        net_config['mixer'] = nature_conv_body(d_hidden=64, d_output=None, n_gen=4),

    return net_config 

ddpg_conv_body = namedtuple('ddpg_conv_body', ' '.join(fields))
ddpg_conv_body.__new__.__defaults__ = defaults # python 3.6
def DDPGConvBody_config(in_channels, feature_dim, mixer=False):
    net_config = {
            'conv1': ddpg_conv_body(d_input=in_channels, d_output=32, kernel=3),
            'conv2': ddpg_conv_body(d_input=32, d_output=32, kernel=3),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 2
            }
    if mixer:
        net_config['mixer'] = ddpg_conv_body(d_hidden=64, d_output=None, n_gen=2),

    return net_config 


toy_fc_body = namedtuple('toy_fc_body', ' '.join(fields))
toy_fc_body.__new__.__defaults__ = defaults # python 3.6
def ToyFCBody_config(d_state, d_hidden, gate, mixer=False):
    net_config = {}
    net_config['fc1'] = toy_fc_body(d_input=d_state, d_output=d_hidden[0], z_dim=24)
    for i in range(1, len(d_hidden)):
        net_config['fc{}'.format(i+1)] = toy_fc_body(d_input=d_hidden[i-1], d_output=d_hidden[i], z_dim=24)
    
    net_config['z_dim'] = 24
    net_config['s_dim'] = 24
    net_config['n_gen'] = len(d_hidden)

    return net_config 


fc_body = namedtuple('fc_body', ' '.join(fields))
fc_body.__new__.__defaults__ = defaults # python 3.6
def FCBody_config(d_state, d_hidden, gate, mixer=False):
    net_config = {}
    net_config['fc1'] = fc_body(d_input=d_state, d_output=d_hidden[0], z_dim=24)
    for i in range(1, len(d_hidden)):
        net_config['fc{}'.format(i+1)] = fc_body(d_input=d_hidden[i-1], d_output=d_hidden[i], z_dim=24)
    if mixer:
        net_config['mixer'] = fc_body(d_hidden=32, d_output=None, n_gen=len(d_hidden))
    net_config['z_dim'] = 24
    net_config['s_dim'] = 32
    net_config['n_gen'] = len(d_hidden)

    return net_config 

two_fc_body_action = namedtuple('two_fc_body_action', ' '.join(fields))
two_fc_body_action.__new__.__defaults__ = defaults # python 3.6
def TwoLayerFCBodyWithAction_config(d_state, d_action, d_hidden, gate, mixer=False):
    net_config = {
            'fc1': two_fc_body_action(d_input=d_state, d_output=d_hidden[0]),
            'fc2': two_fc_body_action(d_input=d_hidden[0]+d_action, d_output=d_hidden[1]),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 2
            }
    if mixer:
        net_config['mixer'] = two_fc_body_action(d_hidden=64, d_output=None, n_gen=2),

    return net_config 

one_fc_body_action = namedtuple('one_fc_body_action', ' '.join(fields))
one_fc_body_action.__new__.__defaults__ = defaults # python 3.6
def OneLayerFCBodyWithAction_config(d_state, d_action, d_hidden, gate, mixer=False):
    net_config = {
            'fc_s': one_fc_body_action(d_input=d_state, d_output=d_hidden),
            'fc_a': one_fc_body_action(d_input=d_action, d_output=d_hidden),
            'z_dim': 64,
            's_dim': 64,
            'n_gen': 2
            }
    if mixer:
        net_config['mixer'] = one_fc_body_action(d_hidden=64, d_output=None, n_gen=2),

    return net_config 

dummy_body = namedtuple('dummy_body', ' '.join(fields))
dummy_body.__new__.__defaults__ = defaults # python 3.6
def Dummy_config():
    net_config = {
            'fc_s': dummy_body(d_input=0, d_output=0),
            'fc_a': dummy_body(d_input=0, d_output=0),
            'z_dim': 0,
            's_dim': 0,
            'n_gen': 0
            }
    return net_config 


