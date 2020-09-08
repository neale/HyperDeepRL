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
from tqdm import tqdm

class DQNThompsonActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()
        self.sigterm = False
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
        q_mean = q_values.mean()
        q_var = q_values.var()
        q_values = to_np(q_values).flatten()
        if self._total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self._task.step([action])
        if done:
            self.update = True,
            self.update_steps = self.episode_steps
            self.episode_steps = 0
        if info[0]['terminate'] == True:
            self.sigterm = True
            self.close()
        info[0]['q_var'] = q_var
        info[0]['q_mean'] = q_mean
        info[0]['p_var'] = 0.
        entry = [self._state[0], action, reward[0], next_state[0], int(done[0]), info]
        self._total_steps += 1
        self._state = next_state
        self.episode_steps += 1
        return entry


class DQN_Ensemble_Thompson_Agent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()
        self.actor = DQNThompsonActor(config)
        
        self.network_ensemble = []
        self.prior_network_ensemble = []
        self.target_network_ensemble = []
        self.optimizer_ensemble = []
        self.replay_ensemble = []
        for _ in range(config.ensemble_len):
            network = config.network_fn()
            prior_network = config.network_fn()
            network.share_memory()
            target_network = config.network_fn()
            target_network.load_state_dict(network.state_dict())
            self.network_ensemble.append(network)
            self.prior_network_ensemble.append(prior_network)
            self.target_network_ensemble.append(target_network)
            self.optimizer_ensemble.append(config.optimizer_fn(network.parameters())) 
            self.replay_ensemble.append(config.replay_fn())
        
        self.head = np.random.choice(config.ensemble_len, 1)[0]
        self.actor.set_network(self.network_ensemble[self.head])

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay_ensemble[0].batch_size)
        print (self.network_ensemble[0], 'X {}'.format(config.ensemble_len))


    def close(self):
        for i in range(self.config.ensemble_len):
            close_obj(self.replay_ensemble[i])
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network_ensemble[self.head](state)
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config
        if self.actor.sigterm:
            self.total_steps = config.max_steps
            self.close()
        
        beta = config.prior_scale
        network = self.network_ensemble[self.head]
        prior_network = self.prior_network_ensemble[self.head]
        target_network = self.target_network_ensemble[self.head]
        self.actor.set_network(network)
        self.actor.set_prior_network(prior_network)
        transitions = self.actor.step()
        experiences = []
        for state, action, reward, next_state, done, info in transitions:
            self.record_online_return(info)
            self.total_steps += 1
            reward = config.reward_normalizer(reward)
            experiences.append([state, action, reward, next_state, done])

        mask = np.random.uniform(0, 1, size=1)
        if mask < config.replay_mask_prob:
            for i in range(config.ensemble_len):
                self.replay_ensemble[i].feed_batch(experiences)
        else:
            self.replay_ensemble[self.head].feed_batch(experiences)

        if self.total_steps > self.config.exploration_steps and self.actor.update:
            for update in tqdm(range(self.actor.update_steps), desc='SGD updates'):
                experiences = self.replay_ensemble[self.head].sample()
                states, actions, rewards, next_states, terminals = experiences
                states = self.config.state_normalizer(states)
                next_states = self.config.state_normalizer(next_states)
                q_next = target_network(next_states).detach()  # [particles, batch, action]
                prior_next = prior_network(next_states).detach()
                q_next = q_next + beta * prior_next
                if self.config.double_q:
                    q = network(next_states) + beta * prior_network(next_states).detach()
                    best_actions = torch.argmax(q, dim=-1)  # get best action  [batch]
                    q_next = q_next[self.batch_indices, best_actions]
                else:
                    q_next = q_next.max(1)[0]
                terminals = tensor(terminals)
                rewards = tensor(rewards)
                q_next = self.config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                actions = tensor(actions).long()
                q = network(states) + beta * prior_network(states).detach()
                q = q[self.batch_indices, actions]

                loss = (q_next - q).pow(2).mul(0.5).mean()
                self.optimizer_ensemble[self.head].zero_grad()
                loss.backward()
                if self.config.gradient_clip:
                    nn.utils.clip_grad_norm_(network.parameters(), self.config.gradient_clip)
                with config.lock:
                    self.optimizer_ensemble[self.head].step()

            target_network.load_state_dict(network.state_dict())
            self.actor.update = False
            self.head = np.random.choice(config.ensemble_len, 1)[0]
