#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import copy
from .network_utils import *
from .network_bodies import *
from .hyper_bodies import * 
from .hypernetwork_ops import *
from ..utils.hypernet_heads_defs import *
from ..component.samplers import *
from .critics import *

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
    def __init__(self, action_dim, body, hidden, dist, particles, generate_body=False):
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
        self.noise_sampler = NoiseSampler(dist, self.z_dim, self.particles)
        self.generate_body = generate_body

        self.sample_model_seed()
        self.to(Config.DEVICE)
    
    def sample_model_seed(self, return_seed=False, aux_noise=1e-6):
        sample_z = self.noise_sampler.sample(aux_noise).to(Config.DEVICE)
        sample_z = sample_z.unsqueeze(0).repeat(self.features.config['n_gen'], 1, 1)
        model_seed = {
            'features_z': sample_z,
            'value_z': sample_z[0],
            'advantage_z': sample_z[0],
        }
        if return_seed:
            return model_seed
        else:
            self.model_seed = model_seed

    def sweep_samples(self):
        samples = self.noise_sampler.sweep_samples().to(Config.DEVICE)
        samples = samples.unsqueeze(0).repeat(self.features.config['n_gen'], 1, 1)
        return {
                'features_z': samples,
                'value_z': samples[0],
                'advantage_z': samples[0],
            }

    def set_model_seed(self, seed):
        self.model_seed = seed
    
    def generate_theta(self, seed):
        if self.generate_body:
            self.features.generate_theta(seed['features_z'])
            theta_body = self.features.theta
        theta_value = self.fc_value(seed['value_z'])
        theta_adv = self.fc_advantage(seed['advantage_z'])

        self.theta_value_len = theta_value.size(1)
        self.theta_adv_len = theta_adv.size(1)

        self.theta_all = torch.cat((theta_body, theta_value, theta_adv), -1)

    def get_theta(self, layer):
        value = self.fc_value.d_output * self.fc_value.d_input + self.fc_value.d_output
        adv = self.fc_advantage.d_output * self.fc_advantage.d_input + self.fc_advantage.d_output
        val_len = body_len + self.theta_value_len
        if self.generate_body:
            body_len = self.features.body_params_len
        else:
            body_len = 0

        theta = None
        if layer == 'value':
            theta = self.theta_all[:, body_len:val_len]
        elif layer == 'adv':
            theta = self.theta_all[:, val_len:]
        elif layer == 'all':
            theta = self.theta_all
        return theta


    def forward(self, x, seed=None, theta=False, to_numpy=False):
        if not isinstance(x, torch.cuda.FloatTensor):
            x = tensor(x)
        if x.shape[0] == 1 and x.shape[1] == 1: ## dm_env returns one too many dimensions
            x = x[0]
        phi = self.body(x, seed, theta)
        print ('body', phi.shape)
        return self.head(phi, seed, theta)

    def body(self, x=None, seed=None, theta=False):
        if not isinstance(x, torch.cuda.FloatTensor):
            x = tensor(x)
        z = seed if seed != None else self.model_seed
        if self.generate_body:
            return self.features(x)
        else:
            return self.features(x, z['features_z'], theta)

    def head(self, phi, seed=None, theta=False):
        z = seed if seed != None else self.model_seed
        if theta:
            value = self.fc_value.evaluate(phi, self.get_theta('value'))
            advantage = self.fc_advantage.evaluate(phi, self.get_theta('adv'))
        else:
            value = self.fc_value.evaluate(
                    phi,
                    self.fc_value(z['value_z']))
            advantage = self.fc_advantage.evaluate(
                    phi,
                    self.fc_advantage(z['advantage_z']))
        q = value.expand_as(advantage) + (advantage - advantage.mean(-1, keepdim=True).expand_as(advantage))
        return q

    def predict_action(self, x, pred, theta=False, to_numpy=False):
        x = tensor(x)
        q = self(x, theta=theta)
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

class DuelingHyperHead(nn.Module, BaseNet):
    def __init__(self, action_dim, body, hidden, dist, particles, critic_hidden):
        super(DuelingHyperHead, self).__init__()
        self.mixer = False
        
        self.config = DuelingNet_config(body.feature_dim, action_dim)
        self.config['fc_value'] = self.config['fc_value']._replace(d_hidden=hidden)
        self.config['fc_advantage'] = self.config['fc_advantage']._replace(d_hidden=hidden)
        self.fc_value = LinearGenerator(self.config['fc_value']).cuda()
        self.fc_advantage = LinearGenerator(self.config['fc_advantage']).cuda()
        self.features = body
        self.critic_hidden = critic_hidden
        
        self.s_dim = self.config['s_dim']
        self.z_dim = self.config['z_dim']
        self.n_gen = self.config['n_gen']
        self.particles = particles
        self.noise_sampler = NoiseSampler(dist, self.z_dim, self.particles)

        self.sample_model_seed()
        self.configure_critic()
        self.to(Config.DEVICE)
    
    def configure_critic(self):
        input_dim = output_dim = self.generate_theta(store=False).shape[1]
        self.critic = BasicCritic(input_dim, output_dim, self.critic_hidden)
        self.critic = self.critic.to(Config.DEVICE)
        self.critic_init = copy.deepcopy(self.critic).to(Config.DEVICE)

    def sample_model_seed(self, return_seed=False, aux_noise=0., sweep=False):
        if sweep:
            sample_z = self.noise_sampler.sweep_samples(aux_noise).to(Config.DEVICE)
        else:
            sample_z = self.noise_sampler.sample(aux_noise).to(Config.DEVICE)
        model_seed = {
            'value_z': sample_z,
            'advantage_z': sample_z,
        }
        if return_seed or sweep:
            return model_seed
        else:
            self.model_seed = model_seed

    def set_model_seed(self, seed):
        self.model_seed = seed
    
    def generate_theta(self, seed=None, store=True):
        """Generates and optionally stores new parameters for value
            and advantage heads
        """
        if seed is None:
            seed = self.model_seed
        theta_value = self.fc_value(seed['value_z'])
        theta_adv = self.fc_advantage(seed['advantage_z'])
        
        if store is True:
            self.theta_value_len = theta_value.size(1)
            self.theta_adv_len = theta_adv.size(1)
            self.theta_all = torch.cat((theta_value, theta_adv), -1)
        else:
            return torch.cat((theta_value, theta_adv), -1)

    def get_theta(self, layer):
        """Retrieve earlier computed parameter values for value and advantage
            heads. Does not compute new parameters
        """
        value = self.fc_value.d_output * self.fc_value.d_input + self.fc_value.d_output
        adv = self.fc_advantage.d_output * self.fc_advantage.d_input + self.fc_advantage.d_output
        val_len = self.theta_value_len

        theta = None
        if layer == 'value':
            theta = self.theta_all[:, :val_len]
        elif layer == 'adv':
            theta = self.theta_all[:, val_len:]
        elif layer == 'all':
            theta = self.theta_all
        return theta

    def input_as_ensemble(self, x, expand_dim):
        d = [1] * x.dim()
        if x.dim() > 2:
            x = x.transpose(0, 1)
            x = x.repeat(expand_dim, *d[:-1])
        else:
            x = x.unsqueeze(0).repeat(expand_dim, *d)
        return x

    def dueling(self, v, a):
        return v.expand_as(a) + (a - a.mean(-1, keepdim=True).expand_as(a))
        

    def forward(self, x, seed=None, to_numpy=False):
        """Computes Q values for some input x
        
            Args:
                x (torch.tensor): input state
                seed (dict): optional model seed from which to generate 
                    layer parameters for value and advantage heads
        """

        if not isinstance(x, torch.cuda.FloatTensor):
            x = tensor(x)
        if x.shape[0] == 1 and x.shape[1] == 1: ## dm_env returns one too many dimensions
            x = x[0]
        phi = self.body(x)
        phi = self.input_as_ensemble(phi, expand_dim=self.particles)
        return self.head(phi)

    def body(self, x):
        """Computes features from input state"""
        if not isinstance(x, torch.cuda.FloatTensor):
            x = tensor(x)
        return self.features(x)

    def head(self, phi):
        """Compute output Q values according to Dueling DQN formulation: 
            Q(s, a) + [A(s, a) - mean( A(s, a) )]
        """
        value = self.fc_value.evaluate(phi, self.get_theta('value'))
        advantage = self.fc_advantage.evaluate(phi, self.get_theta('adv'))
        return self.dueling(value, advantage)

    def forward_with_seed_or_theta(self, x, seed, theta=None):
        """performs one forward pass with a different model seed than the 
            one stored. Useful for resampling
        """
        if theta is None:
            theta = self.generate_theta(seed, store=False)
        value = self.fc_value.d_output * self.fc_value.d_input + self.fc_value.d_output
        adv = self.fc_advantage.d_output * self.fc_advantage.d_input + self.fc_advantage.d_output
        val_len = self.theta_value_len
        theta_v = theta[:, :val_len]
        theta_adv = theta[:, val_len:]
        
        phi = self.body(x)
        phi = self.input_as_ensemble(phi, expand_dim=seed['value_z'].shape[0])
        value = self.fc_value.evaluate(phi, theta_v)
        advantage = self.fc_advantage.evaluate(phi, theta_adv)
        return self.dueling(value, advantage)
        
    def predict_action(self, x, pred, theta=False, to_numpy=False):
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

class DuelingHyperNetFx(nn.Module, BaseNet):
    def __init__(self, action_dim, body, hidden, dist, particles):
        super(DuelingHyperNetFx, self).__init__()
        self.mixer = False
        
        self.config = DuelingNet_config(body.feature_dim, action_dim)
        self.config['fc_value'] = self.config['fc_value']._replace(d_hidden=hidden)
        self.config['fc_advantage'] = self.config['fc_advantage']._replace(d_hidden=hidden)
        self.fc_value = LinearGeneratorFx(self.config['fc_value']).cuda()
        self.fc_advantage = LinearGeneratorFx(self.config['fc_advantage']).cuda()
        self.features = body
        
        self.s_dim = self.config['s_dim']
        self.z_dim = self.config['z_dim']
        self.n_gen = self.config['n_gen'] + self.features.config['n_gen'] + 1
        self.particles = particles
        self.noise_sampler = NoiseSampler(dist, self.z_dim, self.particles)
        self.sample_model_seed()
        self.to(Config.DEVICE)
    
    def sample_model_seed(self, return_seed=False, aux_noise=1e-6):
        sample_z = self.noise_sampler.sample(aux_noise).to(Config.DEVICE)
        sample_z = sample_z.unsqueeze(0).repeat(self.features.config['n_gen'], 1, 1)
        model_seed = {
            'features_z': sample_z,
            'value_z': sample_z[0],
            'advantage_z': sample_z[0],
        }
        if return_seed:
            return model_seed
        else:
            self.model_seed = model_seed

    def sweep_samples(self):
        samples = self.noise_sampler.sweep_samples().to(Config.DEVICE)
        samples = samples.unsqueeze(0).repeat(self.features.config['n_gen'], 1, 1)
        return {
                'features_z': samples,
                'value_z': samples[0],
                'advantage_z': samples[0],
            }

    def set_model_seed(self, seed):
        self.model_seed = seed

    def forward(self, x, seed=None, to_numpy=False):
        if not isinstance(x, torch.cuda.FloatTensor):
            x = tensor(x)
        if x.shape[0] == 1 and x.shape[1] == 1: ## dm_env returns one too many dimensions
            x = x[0]
        phi = self.body(x, seed)
        return self.head(phi, seed)

    def body(self, x=None, seed=None, theta=None):
        if not isinstance(x, torch.cuda.FloatTensor):
            x = tensor(x)
        z = seed if seed != None else self.model_seed
        return self.features(x, z['features_z'], theta)

    def head(self, phi, seed=None, theta_v=None, theta_a=None):
        z = seed if seed != None else self.model_seed
        value = self.fc_value(z['value_z'], phi, theta_v)
        advantage = self.fc_advantage(z['advantage_z'], phi, theta_a)
        q = value.expand_as(advantage) + (advantage - advantage.mean(-1, keepdim=True).expand_as(advantage))
        return q

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
