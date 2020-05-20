#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
from .network_bodies import *
from .hyper_bodies import * 
from .hypernetwork_ops import *
from ..utils.hypernet_heads_defs import *
from ..component.samplers import *

class VanillaHyperNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(VanillaHyperNet, self).__init__()
        self.mixer = False
        self.config = VanillaNet_config(body.feature_dim, output_dim)
        self.fc_head = LinearGenerator(self.config['fc_head'])
        self.body = body
        self.to(Config.DEVICE)

    def sample_model_seed(self):
        if not self.mixer:
            self.model_seed = {
                    'fc_head_z': torch.rand(self.fc_head.config['n_gen'], particles, self.z_dim).to(Config.DEVICE)
            }
        else:
            self.model_seed = torch.rand(particles, self.s_dim)
    
    def forward(self, x, z=None):
        phi = self.body(tensor(x, z))
        y = self.fc_head(z[0], phi)
        return y


class DuelingHyperNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body, hidden, dist, particles,dist_scale=1e-5):
        super(DuelingHyperNet, self).__init__()
        self.mixer = False
        
        self.config = DuelingNet_config(body.feature_dim, action_dim)
        self.config['fc_value'] = self.config['fc_value']._replace(d_hidden=hidden)
        self.config['fc_advantage'] = self.config['fc_advantage']._replace(d_hidden=hidden)
        self.fc_value = LinearGenerator(self.config['fc_value']).cuda()
        self.fc_advantage = LinearGenerator(self.config['fc_advantage']).cuda()
        self.features = body
        
        self.s_dim = self.config['s_dim']
        self.z_dim = self.config['z_dim']
        self.n_gen = self.config['n_gen'] + self.features.config['n_gen'] + 1
        self.particles = particles
        self.noise_sampler = NoiseSampler(
            dist, self.z_dim, aux_scale=dist_scale, particles=self.particles)
            # dist, self.z_dim, particles=self.particles)
        self.sample_model_seed()
        self.to(Config.DEVICE)
    
    def sample_model_seed(self, return_seed=False):
        sample_z = self.noise_sampler.sample().to(Config.DEVICE)
        sample_z = sample_z.unsqueeze(0).repeat(self.features.config['n_gen'], 1, 1)
        # sample_z = sample_z.unsqueeze(0).unsqueeze(0).repeat(
        #     self.features.config['n_gen'], self.particles, 1)
        model_seed = {
            'features_z': sample_z,
            'value_z': sample_z[0],
            'advantage_z': sample_z[0],
            'mdp_z': sample_z[0],
            'reward_z': sample_z[0],
        }
        if return_seed:
            return model_seed
        else:
            self.model_seed = model_seed

    def sweep_samples(self, particles=None):
        s = self.noise_sampler.sweep_samples(particles).to(Config.DEVICE)
        s = s.unsqueeze(0).repeat(self.features.config['n_gen'], 1, 1)
        return {
                'features_z': s,
                'value_z': s[0],
                'advantage_z': s[0]
                }

    def set_model_seed(self, seed):
        self.model_seed = seed

    def forward(self, x, seed=None, to_numpy=False, ensemble_input=False):
        phi = self.body(x, seed=seed, ensemble_input=ensemble_input)
        return self.head(phi, seed=seed)

    def body(self, x=None, seed=None, theta=None, ensemble_input=False):
        if not isinstance(x, torch.cuda.FloatTensor):
            x = tensor(x)
        if x.shape[0] == 1 and x.shape[1] == 1: ## dm_env returns one too many dimensions
            x = x[0]
        z = seed if seed != None else self.model_seed
        return self.features(x, z['features_z'], theta, ensemble_input=ensemble_input)

    def head(self, phi, seed=None, theta_v=None, theta_a=None):
        z = seed if seed != None else self.model_seed
        value = self.fc_value(z['value_z'], phi, theta_v)
        advantage = self.fc_advantage(z['advantage_z'], phi, theta_a)
        q = value.expand_as(advantage) + (advantage - advantage.mean(-1, keepdim=True).expand_as(advantage))
        return q

    def sample_model(self, component='q', seed=None):
        seed = seed if seed is not None else self.model_seed
        param_sets = []
        if component == 'q':
            #param_sets.extend(self.features(z=seed['features_z']))
            return self.fc_value(z=seed['value_z']), self.fc_advantage(z=seed['advantage_z'])

            param_sets.extend(self.fc_value(z=seed['value_z']))
            param_sets.extend(self.fc_advantage(z=seed['advantage_z']))
        return param_sets

    def predict_action(self, x, pred, to_numpy=False):
        x = tensor(x)
        q = self(x)
        if pred == 'max':
            max_q, max_q_idx = q.max(-1)  # max over q values
            max_actor = max_q.max(0)[1]  # max over particles
            action = q[max_actor].argmax()
        
        elif pred == 'rand':
            idx = np.random.choice(self.particles, 1)[0]
            action = q[idx].max(0)[1]
        
        elif pred == 'mean':
            action_means = q.mean(0)  #[actions]
            action = action_means.argmax()

        if to_numpy:
            action = action.cpu().detach().numpy()
        return action


class MdpHyperNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body, hidden, dist, particles): 
        super(MdpHyperNet, self).__init__()
        self.config = MdpNet_config(body.state_dim, body.feature_dim)
        self.config['fc_mdp'] = self.config['fc_mdp']._replace(
            d_hidden=hidden)
        self.config['fc_reward'] = self.config['fc_reward']._replace(
            d_hidden=hidden)
        self.fc_mdp = LinearGenerator(self.config['fc_mdp']).cuda()
        self.fc_reward = LinearGenerator(self.config['fc_reward']).cuda()
        self.features = body

        self.s_dim = self.config['s_dim']
        self.z_dim = self.config['z_dim']
        self.n_gen = self.config['n_gen'] + self.features.config['n_gen'] + 1
        self.particles = particles
        self.state_dim = body.state_dim
        self.action_dim = action_dim
        self.noise_sampler = NoiseSampler(
            dist, self.z_dim, aux_scale=1e-5, particles=self.particles)
            # dist, self.z_dim, particles=self.particles)
        self.sample_model_seed()
        self.to(Config.DEVICE)

    def sample_model_seed(self, particles=None, return_seed=False):
        sample_z = self.noise_sampler.sample(particles).to(Config.DEVICE)
        sample_z = sample_z.unsqueeze(0).repeat(self.features.config['n_gen'], 1, 1)
        # sample_z = sample_z.unsqueeze(0).unsqueeze(0).repeat(
        #     self.features.config['n_gen'], self.particles, 1) # [n_gen, particles, z_dim]
        model_seed = {
            'features_z': sample_z,
            'mdp_z': sample_z[0],
            'reward_z': sample_z[0],
        }
        if return_seed:
            return model_seed
        else:
            self.model_seed = model_seed

    def sweep_samples(self, particles=None):
        s = self.noise_sampler.sweep_samples(particles).to(Config.DEVICE)
        s = s.unsqueeze(0).repeat(self.features.config['n_gen'], 1, 1)
        return {
                'features_z': s,
                'value_z': s[0],
                'advantage_z': s[0]
                }


    def set_model_seed(self, seed):
        self.model_seed = seed

    def forward(self, x, a, seed=None, to_numpy=False, ensemble_input=False):
        # when ensemble_input=True: both x and a are of the shape [p, batch, d_input]
        phi, x = self.body(x, seed=seed, ensemble_input=ensemble_input) # [p, batch, d_out]
        all_but_last_dims = phi.shape[:-1]
        phi = phi.view(*all_but_last_dims, -1, self.action_dim)
        if ensemble_input:
            a = a.unsqueeze(2).repeat(1, 1, phi.shape[2], 1)
            phi = phi.gather(-1, a).squeeze(-1)
        else:
            batch_indices = range_tensor(phi.shape[1])
            phi = phi[:, batch_indices, :, a]  
            phi = torch.transpose(phi, 0, 1) # [p, bs, feature_dim]
        delta_x, rewards = self.head(phi, seed=seed)
        return delta_x + x, rewards

    def body(self, x=None, seed=None, theta=None, ensemble_input=False):
        if not isinstance(x, torch.cuda.FloatTensor):
            x = tensor(x)
        if x.shape[0] == 1 and x.shape[1] == 1: ## dm_env returns one too many dimensions
            x = x[0]
        z = seed if seed != None else self.model_seed
        return self.features(x, z['features_z'], theta, ensemble_input=ensemble_input)

    def head(self, phi, seed=None):
        z = seed if seed != None else self.model_seed
        delta_x = self.fc_mdp(z['mdp_z'], phi)
        reward = self.fc_reward(z['reward_z'], phi)
        # delta_x, reward = torch.split(mdp_out, [self.state_dim, 1], dim=-1)
        return delta_x, reward
        
    #def sample_model(self, component='q', seed=None):
    #    seed = seed if seed is not None else self.model_seed
    #    param_sets = []
    #    if component == 'q':
    #        #param_sets.extend(self.features(z=seed['features_z']))
    #        return self.fc_value(z=seed['value_z']), self.fc_advantage(z=seed['advantage_z'])

    #        param_sets.extend(self.fc_value(z=seed['value_z']))
    #        param_sets.extend(self.fc_advantage(z=seed['advantage_z']))
    #    return param_sets

class FlatMdpHyperNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body, hidden, dist, particles): 
        super(FlatMdpHyperNet, self).__init__()
        self.config = MdpNet_config(body.state_dim, body.feature_dim)
        self.config['fc_mdp'] = self.config['fc_mdp']._replace(
            d_hidden=hidden)
        self.config['fc_reward'] = self.config['fc_reward']._replace(
            d_hidden=hidden)
        
        self.fc_mdp = FlatLinearGenerator(self.config['fc_mdp']).cuda()
        self.fc_reward = FlatLinearGenerator(self.config['fc_reward']).cuda()
        self.features = body

        self.s_dim = self.config['s_dim']
        self.z_dim = self.config['z_dim']
        self.n_gen = self.config['n_gen'] + self.features.config['n_gen'] + 1
        self.particles = particles
        self.state_dim = body.state_dim
        self.action_dim = action_dim
        self.noise_sampler = NoiseSampler(
            dist, self.z_dim, aux_scale=1e-5, particles=self.particles)
            # dist, self.z_dim, particles=self.particles)
        self.sample_model_seed()
        self.to(Config.DEVICE)

    def sample_model_seed(self, particles=None, return_seed=False):
        sample_z = self.noise_sampler.sample(particles).to(Config.DEVICE)
        sample_z = sample_z.unsqueeze(0).repeat(self.features.config['n_gen'], 1, 1)
        # sample_z = sample_z.unsqueeze(0).unsqueeze(0).repeat(
        #     self.features.config['n_gen'], self.particles, 1) # [n_gen, particles, z_dim]
        model_seed = {
            'features_z': sample_z,
            'mdp_z': sample_z[0],
            'reward_z': sample_z[0],
        }
        if return_seed:
            return model_seed
        else:
            self.model_seed = model_seed

    def sweep_samples(self, particles=None):
        s = self.noise_sampler.sweep_samples(particles).to(Config.DEVICE)
        s = s.unsqueeze(0).repeat(self.features.config['n_gen'], 1, 1)
        return {
                'features_z': s,
                'value_z': s[0],
                'advantage_z': s[0]
                }

    def set_model_seed(self, seed):
        self.model_seed = seed

    """ Generate parameters for the network 
    reward and mdp heads share the body parameters:
    mdp = mdp_head(body(params), params_mdp)
    reward = reward_head(body(params), params_reward)
    but when we take gradients with respect to theta we need to have every op in the graph. 
    i.e. we can't perfrom another operation like cat() after the computation graph
    """

    def generate_theta(self, seed):
        self.features.generate_theta(seed['features_z'])
        body_theta = self.features.theta
        mdp_theta = self.fc_mdp(seed['mdp_z'])
        reward_theta = self.fc_reward(seed['reward_z'])

        self.mdp_len = mdp_theta.size(1)
        self.reward_len = reward_theta.size(1)

        #self.theta = torch.cat((body_theta, mdp_theta, reward_theta), -1)
        self.mdp_theta = torch.cat((body_theta, mdp_theta), -1)
        self.reward_theta = torch.cat((body_theta, reward_theta), -1)

    def get_params(self, layer):
        mdp = self.fc_mdp.d_output * self.fc_mdp.d_input + self.fc_mdp.d_output
        reward = self.fc_reward.d_output * self.fc_reward.d_input + self.fc_reward.d_output
        body = self.features.body_params_len
        theta = None
        if layer == 'mdp':
            theta = self.mdp_theta[:, body:]
        elif layer == 'reward':
            theta = self.reward_theta[:, body:]
        return theta

    def forward(self, x, a, to_numpy=False, ensemble_input=False):
        # when ensemble_input=True: both x and a are of the shape [p, batch, d_input]
        phi, x = self.body(x, ensemble_input=ensemble_input) # [p, batch, d_out]
        all_but_last_dims = phi.shape[:-1]
        phi = phi.view(*all_but_last_dims, -1, self.action_dim)
        if ensemble_input:
            a = a.unsqueeze(2).repeat(1, 1, phi.shape[2], 1)
            phi = phi.gather(-1, a).squeeze(-1)
        else:
            batch_indices = range_tensor(phi.shape[1])
            phi = phi[:, batch_indices, :, a]  
            phi = torch.transpose(phi, 0, 1) # [p, bs, feature_dim]
        delta_x, rewards = self.head(phi)
        return delta_x + x, rewards

    def body(self, x, ensemble_input=False):
        if not isinstance(x, torch.cuda.FloatTensor):
            x = tensor(x)
        if x.shape[0] == 1 and x.shape[1] == 1: ## dm_env returns one too many dimensions
            x = x[0]
        return self.features(x, ensemble_input=ensemble_input)

    def head(self, phi):
        delta_x = self.fc_mdp.evaluate(phi, self.get_params('mdp'))
        reward = self.fc_reward.evaluate(phi, self.get_params('reward'))
        return delta_x, reward
        
