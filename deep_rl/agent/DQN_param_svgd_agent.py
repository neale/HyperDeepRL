#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
# Same #
########
import warnings
warnings.filterwarnings("ignore")
from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *
import torchvision
import torch.autograd as autograd
import sys
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


class DQNParamSVGDActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()
        self.k = np.random.choice(config.particles, 1)[0]
        self.update = False
        self.episode_steps = 0
        self.update_steps = 0
        self.sigterm = False
        self.vis_now = True
        self.ep = 0

    def get_states(self, state):
        x = torch.zeros(len(state[0]), len(state[0]))
        for i in range(len(x)): # number of states
            s = torch.tensor([1 if k <= i else 0 for k in range(len(x))])
            x[i] += s
        return x

    def visualize(self, state):
        states = self.get_states(state)
        samples = self._network.sweep_samples()
        plt.figure(figsize=(12, 15), dpi=200)
        plt.suptitle('Implicit Value Networks - Episode {}'.format(self.ep))
        left_v, right_v, v = [], [], []
        print (len(samples))
        with plt.style.context('seaborn-darkgrid'):
            for i, sample in enumerate(samples):
                value_f = []
                q_vals = self._network(states.cuda(), seed=sample)
                value = q_vals.mean(0)
                left_value = value[:, 0]
                right_value = value[:, 1]
                value = q_vals.mean(0).mean(1)
                left_v.append(left_value.detach().cpu())
                right_v.append(right_value.detach().cpu())
                v.append(value.detach().cpu())
                
                plt.subplot(311)
                plt.plot(range(len(states)), left_value.detach().cpu(), linewidth=.7)
                plt.subplot(312)
                plt.plot(range(len(states)), right_value.detach().cpu(), linewidth=.7)
                plt.subplot(313)
                plt.plot(range(len(states)), (right_value-left_value).detach().cpu(), linewidth=.7)

            plt.subplot(311)
            plt.title('Left Action Value')
            plt.xlabel('State')
            plt.ylabel('Q(s, a)')
            plt.grid(True)
            lvals = np.asarray([t.numpy() for t in left_v]).mean(0)
            plt.plot(range(len(states)), lvals, color='black', linestyle='--', linewidth=2)
            plt.subplot(312)
            plt.xlabel('State')
            plt.ylabel('Q(s, a)')
            plt.grid(True)
            plt.title('Right Action Value')
            rvals = np.asarray([t.numpy() for t in right_v]).mean(0)
            plt.plot(range(len(states)), rvals, color='black', linestyle='--', linewidth=2)
            plt.subplot(313)
            plt.xlabel('State')
            plt.ylabel('Q(s, a)')
            plt.grid(True)
            plt.title('Right Left Difference')
            plt.plot(range(len(states)), rvals-lvals, color='black', linestyle='--', linewidth=2)

        path = 'vis_normal/forever run/chain_len_{}/'.format(len(states))
        os.makedirs(path, exist_ok=True)
        plt.savefig(path+'episode_{}.png'.format(self.ep))
        plt.close('all')
        
        # sys.exit(0)


    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
        config = self.config
        with config.lock:
            state = config.state_normalizer(self._state)
            q_values = self._network(state)
            particle_max = q_values.argmax(-1)
            abs_max = q_values.max(2)[0].argmax()

        q_max = q_values[abs_max]    
        q_max = to_np(q_max).flatten()
        q_var = to_np(q_values.var(0))
        q_mean = to_np(q_values.mean(0))
        q_random = to_np(q_values[self.k])
        
        q_prob = q_values.max(0)[0]
        q_prob = q_prob + q_prob.min().abs() + 1e-8 # to avoid negative or 0 probability of taking an action

        if self.vis_now:
            self.visualize(state)
            self.vis_now = False

        if self._total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
                action = np.random.randint(0, len(q_max))
                actions_log = np.random.randint(0, len(q_max), size=(config.particles, 1))
        else:
            # action = np.argmax(q_max)  # Max Action
            action = np.argmax(q_mean)  # Mean Action
            # action = np.argmax(q_random)  # Random Head Action
            # action = torch.multinomial(q_prob.cpu(), 1, replacement=True).numpy()[0] # Sampled Action
            actions_log = to_np(particle_max)
        
        next_state, reward, done, info = self._task.step([action])
        if config.render and self._task.record_now:
            self._task.render()
        if done:
            self.update = True
            self.update_steps = self.episode_steps
            self.episode_steps = 0
            self.ep += 1
            self.vis_now = True
            self._network.sample_model_seed()
            if self._task.record:
                self._task.record_or_not(info)
                self.k = np.random.choice(config.particles, 1)[0]
        
        # Add Q value estimates to info
        info[0]['q_mean'] = q_mean.mean()
        info[0]['q_var'] = q_var.mean()

        #if info[0]['terminate'] == True:
        #    self.sigterm = True
        #    self.close()
        entry = [self._state[0], action, actions_log, reward[0], next_state[0], int(done[0]), info]
        self._total_steps += 1
        self._state = next_state
        self.episode_steps += 1
        return entry


class DQN_Param_SVGD_Agent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.config.save_config_to_yaml()
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DQNParamSVGDActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.alpha_schedule = BaselinesLinearSchedule(config.alpha_anneal, config.alpha_final, config.alpha_init)
        self.actor.set_network(self.network)

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)
        self.network.sample_model_seed()
        self.target_network.set_model_seed(self.network.model_seed)
        self.head = np.random.choice(config.particles, 1)[0]
        self.save_net_arch_to_file()
        print (self.network)


    def save_net_arch_to_file(self):
        network_str = str(self.network)
        save_fn = '/model_arch.txt'
        save_dir = self.config.tf_log_handle
        with open(save_dir+save_fn, 'w') as f:
            f.write(network_str+'\n\n')

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network.predict_action(state, pred='mean', to_numpy=True)
        action = np.array([action])
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config

        if self.actor.sigterm == True:
            self.close()
            self.total_steps = config.max_steps

        if not torch.equal(self.network.model_seed['value_z'], 
                           self.target_network.model_seed['value_z']):
            self.target_network.set_model_seed(self.network.model_seed)

        transitions = self.actor.step()
        experiences = []
        for state, action, max_actions, reward, next_state, done, info in transitions:
            self.record_online_return(info)
            self.total_steps += 1
            reward = config.reward_normalizer(reward)
            experiences.append([state, action, max_actions, reward, next_state, done])
        self.replay.feed_batch(experiences)
        if self.total_steps == self.config.exploration_steps+1:
            print ('pure exploration finished')

        if self.total_steps > self.config.exploration_steps and self.actor.update:
            for update in tqdm(range(self.actor.update_steps), desc='SGD Q updates'):
                experiences = self.replay.sample()
                states, actions, max_actions, rewards, next_states, terminals = experiences
                states = self.config.state_normalizer(states)
                next_states = self.config.state_normalizer(next_states)
                terminals = tensor(terminals)
                rewards = tensor(rewards)
                
                sample_z = self.network.sample_model_seed(return_seed=True) 
                # all_samples = self.network.sweep_samples()

                p_v, p_a = self.network.sample_model(seed=sample_z) #[layers, particles, layer_dim]
                p_target_v, p_target_a = self.target_network.sample_model(seed=sample_z) #[layers, particles, layer_dim]
                # but we still need to use p to calculate Q values so that we have gradients

                # p_z and p_z_target are the concatenated list of parameters we will use for aSVGD
                pz = []
                for p in [p_v[0], p_v[1], p_a[0], p_a[1]]:
                    pz.append(p.view(config.particles, -1))
                pz = torch.cat(pz, -1) # [particles, n_params]

                pz_t = []
                for p in [p_target_v[0], p_target_v[1], p_target_a[0], p_target_a[1]]:
                    pz_t.append(p.view(config.particles, -1))
                pz_t = torch.cat(pz_t, -1) # [particles, n_params]

                # p and p_target are the jagged lists of parameters for each layer that get passed to the network functions


                # this is going to be bad for the first implementation. Basically in order to take grad(TD, params)
                # I need to use this flattened list in the graph, which means partitioning it as I send it in

                # v_w : [24, 1, 256], 256
                # v_b : [24, 1, 1], 1
                # a_w : [24, 2, 256], 512
                # a_b : [24, 1, 2], 2

                pv = (pz[:, :256].view(24, 1, 256), pz[:, 256:257].view(24, 1, 1))
                pa = (pz[:, 257:769].view(24, 2, 256), pz[:, 769:].view(24, 1, 2))
                
                ptv = (pz_t[:, :256].view(24, 1, 256), pz_t[:, 256:257].view(24, 1, 1))
                pta = (pz_t[:, 257:769].view(24, 2, 256), pz_t[:, 769:].view(24, 1, 2))
                
                ## Get target q values
                q_next_phi = self.target_network.body(next_states, seed=sample_z)
                q_next = self.target_network.head(q_next_phi, seed=sample_z, theta_v=ptv, theta_a=pta).detach()
                #q_next = self.target_network(next_states, seed=sample_z).detach()  # [particles, batch, action]
                
                if self.config.double_q:
                    ## Double DQN
                    q_phi = self.network.body(next_states, seed=sample_z)
                    q = self.network.head(q_phi, seed=sample_z, theta_v=pv, theta_a=pa)
                    # q = self.network(next_states, seed=sample_z)  # [particles, batch, action]
                    best_actions = torch.argmax(q, dim=-1)  # get best action  [particles, batch]
                    q_next = torch.stack([q_next[i, self.batch_indices, best_actions[i]] for i in range(config.particles)])#[p, batch, 1]
                else:
                    q_next = q_next.max(1)[0]
                q_next = self.config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)

                actions = tensor(actions).long()
                max_actions = tensor(max_actions).long()
                
                ## Get main Q values
                phi = self.network.body(states, seed=sample_z)
                q = self.network.head(phi, seed=sample_z, theta_v=pv, theta_a=pa) # [particles, batch, action]
                max_actions = max_actions.transpose(0, 1).squeeze(-1)  # [particles, batch, actions]

                ## define q values with respect to all max actions (q), or the action taken (q_a)
                q_a = torch.gather(q, dim=2, index=actions.unsqueeze(0).unsqueeze(-1).repeat(config.particles, 1, 1)) # :/
                q = torch.stack([q[i, self.batch_indices, max_actions[i]] for i in range(config.particles)]) # [particles, batch]

                alpha = self.alpha_schedule.value(self.total_steps)
                q = q.transpose(0, 1).unsqueeze(-1) # [particles, batch, 1]
                q_a = q_a.transpose(0, 1) # [particles, batch, 1]
                q_next = q_next.transpose(0, 1).unsqueeze(-1)  # [particles, batch, 1]
                
                # Loss functions
                moment1_loss = (q_next.mean(1) - q.mean(1)).pow(2).mul(.5).mean()
                moment2_loss = (q_next.var(1) - q.var(1)).pow(2).mul(.5).mean()
                action_loss = (q_next - q_a).pow(2).mul(0.5)
                sample_loss = (q_next - q).pow(2).mul(0.5) 
                
                # choose which Q to learn with respect to
                if config.svgd_q == 'sample':
                    svgd_q, svgd_q_frozen = pz, pz_frozen
                    td_loss = sample_loss# + moment1_loss# + moment1_loss
                elif config.svgd_q == 'action':
                    td_loss = action_loss# + moment1_loss + moment2_loss

                q_grad = autograd.grad(td_loss.sum(), inputs=pz)[0]
                q_grad = q_grad.unsqueeze(2)  # [particles//2. batch, 1, 1]
                print (q_grad.shape)
                
                # resample for more particles 
                resample_z = self.network.sample_model_seed(return_seed=True) 
                pv, pa = self.network.sample_model(seed=sample_z) #[layers, particles, layer_dim]
                resample_pz = []
                for p in [pv[0], pv[1], pa[0], pa[1]]:
                    resample_pz.append(p.view(config.particles, -1))
                pz_frozen = torch.cat(resample_pz, -1).detach() # [particles, n_params]

                pz_eps = pz + torch.rand_like(pz) * 1e-8
                pz_frozen_eps = pz_frozen + torch.rand_like(pz_frozen) * 1e-8
                
                kappa, grad_kappa = batch_rbf_xy(pz_frozen_eps, pz_eps) 
                kappa = kappa.unsqueeze(-1)
                
                kernel_logp = torch.matmul(kappa.detach(), q_grad.mean(1, keepdim=True)) # [n, 1]
                svgd = (kernel_logp + alpha * grad_kappa).mean(1).squeeze(0) # [n, theta]
                
                self.optimizer.zero_grad()
                autograd.backward(pz, grad_tensors=svgd.detach())
                
                for param in self.network.parameters():
                    if param.grad is not None:
                        param.grad.data *= 1./config.particles
                
                if self.config.gradient_clip: 
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)

                with config.lock:
                    self.optimizer.step()
                self.logger.add_scalar('td_loss', td_loss.mean(), self.total_steps)
                self.logger.add_scalar('grad_kappa', grad_kappa.mean(), self.total_steps)
                self.logger.add_scalar('kappa', kappa.mean(), self.total_steps)
            
                #if self.total_steps / self.config.sgd_update_frequency % \
                #        self.config.target_network_update_freq == 0:
                #     self.target_network.load_state_dict(self.network.state_dict())
            self.target_network.load_state_dict(self.network.state_dict())
            self.actor.update = False 
