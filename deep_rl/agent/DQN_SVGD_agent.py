#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
# Same #
########

from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *
import torchvision
import torch.autograd as autograd
import sys
from tqdm import tqdm

class DQNSVGDActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()
        self.k = np.random.choice(config.particles, 1)[0]

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
        config = self.config
        with config.lock:
            state = config.state_normalizer(self._state)
            q_values = self._network(state)
            particle_max = q_values.argmax(-1)
            abs_max = q_values.max(2)[0].argmax()
        
        q_var = q_values.var(0)
        q_mean = q_values.mean(0)
        q_random = to_np(q_values[self.k])

        posterior_z = self._network.sweep_samples()
        posterior_q = self._network(state, seed=posterior_z)

        if np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, 3)
            actions_log = np.random.randint(-2, -1, size=(config.particles, 1))
        else:
            action = torch.argmax(q_mean).item()  # Mean Action
            #action = np.argmax(q_random)  # Random Head Action
            actions_log = to_np(particle_max)
        
        next_state, reward, done, info = self._task.step([action])
        if done:
            self._network.sample_model_seed()
            self.k = np.random.choice(config.particles, 1)[0]
            if self._task.record:
                self._task.record_or_not(info)
        
        # Add Q value estimates to info
        info[0]['q_mean'] = q_mean.mean()
        info[0]['q_var'] = q_var.mean()
        info[0]['p_var'] = posterior_q.var(0).mean()

        entry = [self._state[0], action, actions_log, reward[0], next_state[0], int(done[0]), info]
        self._total_steps += 1
        self._state = next_state
        return entry


class DQN_SVGD_Agent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.config.save_config_to_yaml()
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DQNSVGDActor(config)

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

        if self.total_steps > self.config.exploration_steps:
            experiences = self.replay.sample()
            states, actions, max_actions, rewards, next_states, terminals = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)
            terminals = tensor(terminals)
            rewards = tensor(rewards)
            
            sample_z = self.network.sample_model_seed(return_seed=True, aux_noise=1e-6) 
            ## Get target q values
            q_next = self.target_network(next_states, seed=sample_z).detach()  # [particles, batch, action]
            if self.config.double_q:
                ## Double DQN
                q = self.network(next_states, seed=sample_z)  # [particles, batch, action]
                best_actions = torch.argmax(q, dim=-1)  # get best action  [particles, batch]
                q_next = torch.stack([q_next[i, self.batch_indices, best_actions[i]] for i in range(config.particles)])#[p, batch, 1]
            else:
                q_next = q_next.max(1)[0]
            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)

            actions = tensor(actions).long()
            max_actions = tensor(max_actions).long()

            if self.config.max_rand:
                n_rand = int(len(actions) * config.max_random_action_prob())
                rand_idx = np.random.randint(len(actions), size=(n_rand))
                for idx in rand_idx:
                    max_actions[idx] = torch.randint(0, 3, size=max_actions[idx].shape).long()
                max_actions = max_actions.long()

            ## Get main Q values
            phi = self.network.body(states, seed=sample_z)
            q = self.network.head(phi, seed=sample_z) # [particles, batch, action]
           
            max_actions = max_actions.transpose(0, 1).squeeze(-1)  # [particles, batch, actions]

            ## define q values with respect to all max actions (q), or the action taken (q_a)
            a_index = actions.unsqueeze(0).unsqueeze(-1).repeat(config.particles, 1, 1) # :/
            q_a = torch.gather(q, dim=2, index=a_index)
            max_index = [q[i, self.batch_indices, max_actions[i]] for i in range(config.particles)] # [particles, batch]
            q = torch.stack(max_index)

            alpha = self.alpha_schedule.value(self.total_steps)
            q = q.unsqueeze(-1) # [particles, batch, 1]
            q_next = q_next.unsqueeze(-1)  # [particles, batch, 1]
            
            qi, qj = torch.split(q, self.config.particles//2, dim=0)  # [batch, particles//2, 1]
            qi_next, qj_next = torch.split(q_next, self.config.particles//2, dim=0) # [batch, particles/2, 1]
            qi_actions, qj_actions = torch.split(q_a, self.config.particles//2, dim=0)  # [batch, particles//2, 1]

            # Loss functions
            moment1_loss = (qj_next.mean(1) - qj.mean(1)).pow(2).mul(.5).mean()
            moment2_loss = (qj_next.var(1) - qj.var(1)).pow(2).mul(.5).mean()
            action_loss = (qj_next - qj_actions).pow(2).mul(0.5)
            sample_loss = (qj_next - qj).pow(2).mul(0.5) 

            # choose which Q to learn with respect to
            if config.svgd_q == 'sample':
                qi, qj = qi, qj
                td_loss = sample_loss# + moment1_loss + moment1_loss
            elif config.svgd_q == 'action':
                qi, qj = qi_actions, qj_actions
                td_loss = action_loss# + moment1_loss + moment2_loss

            q_grad = autograd.grad(td_loss.sum(), inputs=qj)[0]  # fix for j
            q_grad = q_grad.unsqueeze(2)
            
            qi_eps = qi + torch.rand_like(qi) * 1e-8
            qj_eps = qj + torch.rand_like(qj) * 1e-8
            kappa, grad_kappa = batch_rbf_old(qj_eps, qi_eps) 
            
            p_ref = kappa.shape[0]
            #logp_grad = torch.einsum('ij, ijk -> jkl', kappa.unsqueeze(-1), q_grad) / p_ref # [n, 1]
            logp_grad = torch.matmul(kappa, q_grad)# [n, 1]
            grad_out = (logp_grad + alpha * grad_kappa).mean(1) / p_ref # [n, theta]
            
            self.optimizer.zero_grad()
            autograd.backward(qi, grad_tensors=grad_out.detach())
            
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
        
            if self.total_steps / self.config.sgd_update_frequency % \
                    self.config.target_network_update_freq == 0:
                 self.target_network.load_state_dict(self.network.state_dict())
