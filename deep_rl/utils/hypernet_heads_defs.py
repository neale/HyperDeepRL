from collections import namedtuple

# If using python 3.7+, use this template for each nametuple.
# The old way should still work though
# gen_config = namedtuple('layer', fields)
# defaults = (100, 3, False, 64, 64, 'relu', None, 6) # python 3.7

fields = ('d_output', 'd_input', 'd_hidden', 'kernel',
          'bias', 's_dim', 'z_dim', 'act', 'act_out', 'n_gen')
defaults = (64, 3, False, 64, 64, 'relu', None, 1)  # python 3.6

vanilla_net = namedtuple('vanilla_net', ' '.join(fields))
vanilla_net.__new__.__defaults__ = defaults  # python 3.6
def VanillaNet_config(feature_dim, output_dim, mixer=False):
    net_config = {
        'fc_head': vanilla_net(d_input=feature_dim, d_output=output_dim),
        'z_dim': 64,
        's_dim': 64,
        'n_gen': 1
    }
    if mixer:
        net_config['mixer'] = vanilla_net(d_hidden=64, d_output=None, n_gen=1)
    return net_config


toy_dueling_net = namedtuple('toy_dueling_net', ' '.join(fields))
toy_dueling_net.__new__.__defaults__ = defaults  # python 3.6
def ToyDuelingNet_config(feature_dim, action_dim, mixer=False):
    net_config = {
        'fc_value': toy_dueling_net(d_input=feature_dim, d_output=1, z_dim=32),
        'fc_advantage': toy_dueling_net(d_input=feature_dim, d_output=action_dim, z_dim=32),
        'z_dim': 32,
        's_dim': 32,
        'n_gen': 2
    }
    if mixer:
        net_config['mixer'] = toy_dueling_net(
            d_hidden=64, d_output=None, n_gen=2)
    return net_config


dueling_net = namedtuple('dueling_net', ' '.join(fields))
dueling_net.__new__.__defaults__ = defaults  # python 3.6
def DuelingNet_config(feature_dim, action_dim, mixer=False):
    net_config = {
        'fc_value': dueling_net(d_input=feature_dim, d_output=1, z_dim=24),
        'fc_advantage': dueling_net(d_input=feature_dim, d_output=action_dim, z_dim=24),
        'z_dim': 24,
        's_dim': 32,
        'n_gen': 2
    }
    if mixer:
        net_config['mixer'] = dueling_net(d_hidden=64, d_output=None, n_gen=2)
    return net_config


mdp_net = namedtuple('mdp_net', ' '.join(fields))
mdp_net.__new__.__defaults__ = defaults  # python 3.6
def MdpNet_config(input_dim, feature_dim, mixer=False):
    net_config = {
        'fc_mdp': mdp_net(d_input=feature_dim, d_output=input_dim, z_dim=24),
        'fc_reward': mdp_net(d_input=feature_dim, d_output=1, z_dim=24),
        'z_dim': 24,
        's_dim': 32,
        'n_gen': 2
    }
    if mixer:
        net_config['mixer'] = mdp_net(d_hidden=64, d_output=None, n_gen=2)
    return net_config


dynamics_dueling_net = namedtuple('dynamics_dueling_net', ' '.join(fields))
dynamics_dueling_net.__new__.__defaults__ = defaults  # python 3.6
def DynamicsDuelingNet_config(input_dim, feature_dim, action_dim, mixer=False):
    net_config = {
        'fc_value': dynamics_dueling_net(d_input=feature_dim, d_output=1, z_dim=24),
        'fc_advantage': dynamics_dueling_net(
            d_input=feature_dim, d_output=action_dim, z_dim=24),
        'fc_mdp': dynamics_dueling_net(
            d_input=feature_dim + 1, d_output=input_dim + 1, z_dim=24),
        'z_dim': 24,
        's_dim': 32,
        'n_gen': 2
    }
    if mixer:
        net_config['mixer'] = dueling_net(d_hidden=64, d_output=None, n_gen=2)
    return net_config


categorical_net = namedtuple('categorical_net', ' '.join(fields))
categorical_net.__new__.__defaults__ = defaults  # python 3.6
def CategoricalNet_config(feature_dim, action_dim, num_atoms, mixer=False):
    net_config = {
        'fc_categorical': categorical_net(d_input=feature_dim, d_output=action_dim * num_atoms),
        'z_dim': 64,
        's_dim': 64,
        'n_gen': 1
    }
    if mixer:
        net_config['mixer'] = categorical_net(
            d_hidden=64, d_output=None, n_gen=1),
    return net_config


quantile_net = namedtuple('quantile_net', ' '.join(fields))
quantile_net.__new__.__defaults__ = defaults  # python 3.6


def QuantileNet_config(feature_dim, action_dim, num_quantiles, mixer=False):
    net_config = {
        'fc_quantiles': quantile_net(d_input=feature_dim, d_output=action_dim * num_quantiles),
        'z_dim': 64,
        's_dim': 64,
        'n_gen': 1
    }
    if mixer:
        net_config['mixer'] = quantile_net(d_hidden=64, d_output=None, n_gen=1)
    return net_config
