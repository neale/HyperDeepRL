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

class DQN_GFSF_TActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()
        self.update = False
        self.episode_steps = 0
        self.update_steps = 0

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

        posterior_z = self._network.sweep_samples()
        posterior_q = self._network(state, seed=posterior_z)

        if np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, 3)
            actions_log = np.random.randint(-2, -1, size=(config.particles, 1))
        else:
            action = torch.argmax(q_mean).item()  # Mean Action
            actions_log = to_np(particle_max)
        
        next_state, reward, done, info = self._task.step([action])
        if config.render and self._task.record_now:
            self._task.render()
        if done:
            self.update = True
            self.update_steps = self.episode_steps
            self.episode_steps = 0
            self._network.sample_model_seed()
            if self._task.record:
                self._task.record_or_not(info)
        
        # Add Q value estimates to info
        info[0]['q_mean'] = q_mean.mean()
        info[0]['q_var'] = q_var.mean()
        info[0]['p_var'] = posterior_q.var(0).mean()

        entry = [self._state[0], action, actions_log, reward[0], next_state[0], int(done[0]), info]
        self._total_steps += 1
        self._state = next_state
        self.episode_steps += 1
        return entry


class DQN_GFSF_TAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.config.save_config_to_yaml()
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DQN_GFSF_TActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.alpha_schedule = BaselinesLinearSchedule(config.alpha_anneal, config.alpha_final, config.alpha_init)

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)
        
        self.network.sample_model_seed()
        self.network.generate_theta(self.network.model_seed)
        self.target_network.set_model_seed(self.network.model_seed)
        self.target_network.generate_theta(self.network.model_seed)
        
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
            for update in tqdm(range(self.actor.update_steps), desc='SGD Q Updates'):
                experiences = self.replay.sample()
                states, actions, max_actions, rewards, next_states, terminals = experiences
                states = self.config.state_normalizer(states)
                next_states = self.config.state_normalizer(next_states)
                terminals = tensor(terminals)
                rewards = tensor(rewards)
                if np.random.rand() < config.aux_noise_prob():
                    noise = 1e-1
                else:
                    noise = 1e-6
                sample_z = self.network.sample_model_seed(return_seed=True, aux_noise=noise) 
                self.network.generate_theta(sample_z)
                self.target_network.generate_theta(sample_z)
                ## Get target q values
                q_next = self.target_network(next_states, seed=sample_z, theta=True).detach()  # [particles, batch, action]
                if self.config.double_q:
                    ## Double DQN
                    q = self.network(next_states, seed=sample_z, theta=True) # [particles, batch, action]
                    best_actions = torch.argmax(q, dim=-1) # get best action  [particles, batch]
                    #[particles, batch, 1]
                    q_next = torch.stack([q_next[i, self.batch_indices, best_actions[i]] for i in range(config.particles)])
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
                max_actions = max_actions.transpose(0, 1).squeeze(-1)  # [particles, batch, actions]

                ## Get main Q values
                q_vals = self.network(states, seed=sample_z, theta=True) # [particles, batch, action]

                ## define q values with respect to all max actions (q), or the action taken (q_a)
                action_index = actions.unsqueeze(0).unsqueeze(-1).repeat(config.particles, 1, 1)
                q_vals = torch.gather(q_vals, dim=2, index=action_index) # :/
                q_vals = q_vals.squeeze(2)

                #alpha = self.alpha_schedule.value(self.total_steps)
                
                # Loss functions
                params = self.network.get_theta('all')
                
                moment1_loss = (q_next.mean(1) - q_vals.mean(1)).pow(2).mul(.5).mean()
                moment2_loss = (q_next.var(1) - q_vals.var(1)).pow(2).mul(.5).mean()
                td_loss = (q_next - q_vals).pow(2).mul(0.5) 
                q_grad = autograd.grad(td_loss.sum(), inputs=params)[0]  # fix for j
                
                logq_grad = score_func(params)
                grad_out = q_grad - logq_grad
                outputs = params
                
                self.optimizer.zero_grad()
                autograd.backward(outputs, grad_tensors=grad_out.detach())
                
                #for param in self.network.parameters():
                #    if param.grad is not None:
                #        param.grad.data *= 1./config.particles
                
                #if self.config.gradient_clip: 
                #    nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)

                with config.lock:
                    self.optimizer.step()
                self.logger.add_scalar('td_loss', td_loss.mean(), self.total_steps)
                self.logger.add_scalar('score_grad', logq_grad.mean(), self.total_steps)
            self.target_network.load_state_dict(self.network.state_dict())
            self.actor.update = False
