import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import torch
from torch.distributions import Normal

class NChainEnv(gym.Env):
    """n-Chain environment

    This game presents moves along a linear chain of states, with two actions:
     0) forward, which moves along the chain but returns no reward
     1) backward, which returns to the beginning and has a small reward

    The end of the chain, however, presents a large reward, and by moving
    'forward' at the end of the chain this large reward can be repeated.

    At each action, there is a small probability that the agent 'slips' and the
    opposite transition is instead taken.

    The observed state is the current state in the chain (0 to n-1).

    This environment is described in section 6.1 of:
    A Bayesian Framework for Reinforcement Learning by Malcolm Strens (2000)
    http://ceit.aut.ac.ir/~shiry/lecture/machine-learning/papers/BRL-2000.pdf
    """
    def __init__(self, n=10, stochastic=False):
        self.n = n
        self.stochastic = stochastic  # probability of 'slipping' an action
        self.state_int = 1  # Start at beginning of the chain
        self.set_state(self.state_int)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)
        self.seed()
        self.actions = [np.random.choice(2, 1)[0] for i in range(self.n)]
        self.steps = 0
        self.ep_reward = 0.
        self.ep = 1
        self.optim_policy_learned = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def set_state(self, state, encoding='thermometer'):
        if encoding == 'onehot':
            y = torch.eye(self.n)
            self.state = y[state]
        elif encoding == 'thermometer':
            ones = torch.ones(self.n)
            ones[self.state_int+1:] = 0.
            self.state = ones
        self.state = self.state.tolist()
        
    def step(self, action):
        assert self.action_space.contains(action)
        reward = 0.
        if self.stochastic:
            if self.actions[self.state]:
                action = not action  # agent slipped, reverse action taken

        old_state = self.state_int
        # handle actions
        if action == 1:
            if self.state_int >= self.n-1:
                pass
            else:
                self.state_int += 1
                self.set_state(self.state_int)
        else:
            if self.state_int <= 0:
                pass
            else:
                self.state_int -= 1
                self.set_state(self.state_int)

        # handle reward
        if self.state_int == 0:
            reward += 0.001  # small reward for 1st state
        elif self.state_int == self.n - 1:
            reward += 1.  # large reward for end state
        else:
            reward += 0

        self.steps += 1

        self.ep_reward += reward

        info = {'terminate': False}
        
        # print ('state: {}, action: {}, reward: {}, state: {}'.format(old_state, action, reward, self.state))

        if self.steps >= self.n + 9:
            self.ep += 1
            done = True
            # print ('done')
            if self.ep_reward >= 10:  # optimal policy
                self.optim_policy_learned += 1
                print('optimal_policy_episodes', self.optim_policy_learned)
            
            if self.optim_policy_learned >= 100:
                info['terminate'] = True
                self.optim_policy_learned = 0
                print ('Optimal Policy Learned in {} episodes'.format(self.ep))
        else:
            done = False

        
        return self.state, reward, done, info

    def reset(self):
        # print ('reset')
        self.state_int = 1
        self.set_state(self.state_int)
        self.steps = 0
        self.ep_reward = 0.
        return self.state
