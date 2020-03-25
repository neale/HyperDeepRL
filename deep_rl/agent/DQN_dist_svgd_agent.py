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

class DQNDistSVGDActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()

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
        
        ## we want a best action to take, as well as an action for each particle
        #model_action_prob = 1.0
        #if self._total_steps < config.exploration_steps:
        #    model_action_prob = np.random.rand() # 0.5 prob of taking a model steps during exploration
        
        if self._total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
                action = np.random.randint(0, len(q_max))
                actions_log = np.random.randint(0, len(q_max), size=(config.particles, 1))
        else:
            #action = np.argmax(q_max)  # Max Action
            action = np.argmax(q_mean)  # Mean Action
            actions_log = to_np(particle_max)
        
        next_state, reward, done, info = self._task.step([action])
        if config.render and self._task.record_now:
            self._task.render()
        if done:
            self._network.sample_model_seed()
            if self._task.record:
                self._task.record_or_not(info)
        
        # Add Q value estimates to info
        info[0]['q_mean'] = q_mean.mean()
        info[0]['q_var'] = q_var.mean()

        entry = [self._state[0], actions_log, reward[0], next_state[0], int(done[0]), info]
        self._total_steps += 1
        self._state = next_state
        return entry


class DQN_Dist_SVGD_Agent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.config.save_config_to_yaml()
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DQNDistSVGDActor(config)

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
        for state, action, reward, next_state, done, info in transitions:
            self.record_online_return(info)
            self.total_steps += 1
            reward = config.reward_normalizer(reward)
            experiences.append([state, action, reward, next_state, done])
        self.replay.feed_batch(experiences)
        if self.total_steps == self.config.exploration_steps+1:
            print ('pure exploration finished')
        if self.total_steps > self.config.exploration_steps:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)
            terminals = tensor(terminals)
            rewards = tensor(rewards)
            
            ## Get target q values
            q_next = self.target_network(next_states).detach()  # [particles, batch, action]
            if self.config.double_q:
                ## Double DQN
                q = self.network(next_states)  # choose random particle (Q function)  [batch, action]
                best_actions = torch.argmax(q, dim=-1)  # get best action  [batch]
                q_next = torch.stack([q_next[i, self.batch_indices, best_actions[i]] for i in range(config.particles)])
                # q_next = q_next[:, self.batch_indices, best_actions]
            else:
                q_next = q_next.max(1)[0]
            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)
            actions = tensor(actions).long()

            ## Get main Q values
            phi = self.network.body(states)
            q = self.network.head(phi)
            actions = actions.transpose(0, 1).squeeze(-1)
            q = torch.stack([q[i, self.batch_indices, actions[i]] for i in range(config.particles)])
            
            alpha = self.alpha_schedule.value(self.total_steps)
            q = q.transpose(0, 1).unsqueeze(-1)
            q_next = q_next.transpose(0, 1).unsqueeze(-1)
            
            q, q_frozen = torch.split(q, self.config.particles//2, dim=1)  # [batch, particles//2, 1]
            q_next, q_next_frozen = torch.split(q_next, self.config.particles//2, dim=1) # [batch, particles/2, 1]

            q_frozen.detach()
            q_next_frozen.detach()

            td_loss = (q_next - q).pow(2).mul(0.5) #.mean()
            
            q_grad = autograd.grad(td_loss.sum(), inputs=q)[0]
            q_grad = q_grad.unsqueeze(2)  # [particles//2. batch, 1, 1]
            
            q_eps = q + torch.rand_like(q) * 1e-8
            q_frozen_eps = q_frozen + torch.rand_like(q_frozen) * 1e-8

            kappa, grad_kappa = batch_rbf_xy(q_frozen_eps, q_eps) 
            kappa = kappa.unsqueeze(-1)
            
            kernel_logp = torch.matmul(kappa.detach(), q_grad) # [n, 1]
            svgd = (kernel_logp + alpha * grad_kappa).mean(1) # [n, theta]
            
            self.optimizer.zero_grad()
            autograd.backward(q, grad_tensors=svgd.detach())
            
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
