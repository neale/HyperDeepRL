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

class DQNGFSFActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()
        self.k = np.random.choice(self.config.particles, 1)[0]
        self.sigterm = False

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
        config = self.config
        with config.lock:
            state = config.state_normalizer(self._state)
            q_values = self._network(state)
            particle_max = q_values.argmax(-1)

            posterior_z = self._network.sample_model_seed(sweep=True, aux_noise=0)
            posterior_q = self._network.forward_with_seed_or_theta(state, seed=posterior_z)
        
        q_var = q_values.var(0)
        q_mean = q_values.mean(0)

        if np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, 2)
            actions_log = np.random.randint(-2, -1, size=(config.particles, 1))
        else:
            action = torch.argmax(q_values[self.k]).item()  # Mean Action
            actions_log = to_np(particle_max)
        
        next_state, reward, done, info = self._task.step([action])
        if config.render and self._task.record_now:
            self._task.render()
        if done:
            self._network.sample_model_seed()
            self.k = np.random.choice(self.config.particles, 1)[0]
            if self._task.record:
                self._task.record_or_not(info)
        
        # Add Q value estimates to info
        info[0]['q_mean'] = q_mean.mean()
        info[0]['q_var'] = q_var.mean()
        info[0]['p_var'] = posterior_q.var(0).mean()
        
        if info[0]['terminate'] == True:
            self.sigterm = True
            self.close()

        entry = [
                self._state[0],
                action,
                actions_log,
                reward[0],
                next_state[0],
                int(done[0]),
                info]

        self._total_steps += 1
        self._state = next_state
        return entry


class DQN_GFSF_Agent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.config.save_config_to_yaml()
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DQNGFSFActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)
        
        self.network.sample_model_seed()
        self.target_network.set_model_seed(self.network.model_seed)
        
        self.network.generate_theta(self.network.model_seed)
        self.target_network.generate_theta(self.network.model_seed)
        self.theta_init = self.network.theta_all
        
        self.actor.set_network(self.network)

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
        action = self.network.predict_action(state, pred='mean', theta=True, to_numpy=True)
        action = np.array([action])
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config
       
        if not torch.equal(
            self.network.model_seed['value_z'], 
            self.target_network.model_seed['value_z']):
            
            self.target_network.set_model_seed(self.network.model_seed)

        if self.actor.sigterm:
            self.close()
            self.total_steps = config.max_steps

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

        t = self.config.sgd_update_frequency % self.total_steps == 0
        if (self.total_steps > self.config.exploration_steps) and t:
            experiences = self.replay.sample()
            states, actions, max_actions, rewards, next_states, terminals = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)
            terminals = tensor(terminals)
            rewards = tensor(rewards)
            
            # resample 
            z = self.network.sample_model_seed(return_seed=True, aux_noise=0) 
            theta = self.network.generate_theta(z, store=False)
            theta_target = self.target_network.generate_theta(z, store=False)

            ## Get target q values
            q_next = self.target_network.forward_with_seed_or_theta(
                    next_states, z, theta_target).detach()  # [particles, batch, action]
            if self.config.double_q:
                ## Double DQN
                q = self.network.forward_with_seed_or_theta(
                    next_states, z, theta) # [particles, batch, action]
                best_actions = torch.argmax(q, dim=-1) # get best action  [particles, batch]
                #[particles, batch, 1]
                q_next = torch.stack(
                    [q_next[i, self.batch_indices, best_actions[i]] for i in range(
                        config.particles)])
            else:
                q_next = q_next.max(1)[0]
            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)

            actions = tensor(actions).long()
            max_actions = tensor(max_actions).long()

            ## Get main Q values
            q_vals = self.network.forward_with_seed_or_theta(states, z, theta) # [particles, batch, action]
            
            ## define q values with respect to all max actions (q), or the action taken (q_a)
            action_index = actions.unsqueeze(0).unsqueeze(-1).repeat(config.particles, 1, 1)
            q_vals = torch.gather(q_vals, dim=2, index=action_index) # :/
            q_vals = q_vals.squeeze(2)
            
            td_loss = (q_next - q_vals).pow(2).mul(0.5) 
            
            if self.config.use_pushforward:
                inputs = q_vals
            else:
                inputs = theta
            
            logq_grad = autograd.grad(td_loss.sum(), inputs=inputs)[0]  # fix for j
            grad = score_func(inputs)
            grad_out = logq_grad - grad - (theta-self.theta_init).norm().pow(2).mul(.5)
            
            self.optimizer.zero_grad()
            autograd.backward(inputs, grad_tensors=grad_out.detach())
            
            for param in self.network.parameters():
                if param.grad is not None:
                    param.grad.data *= 1./config.particles
            
            if self.config.gradient_clip: 
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)

            with config.lock:
                self.optimizer.step()
            self.logger.add_scalar('td_loss', td_loss.mean(), self.total_steps)
            self.logger.add_scalar('score_grad', logq_grad.mean(), self.total_steps)
            
            if self.total_steps / self.config.sgd_update_frequency % \
                    self.config.target_network_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())