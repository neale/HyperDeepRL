#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
#
# DQN with a generative dynamics model
#
# * In essence, a value-based posterior sampling RL appoach
# * The generative dynamics model captures the agent's uncertainty of the MDP
# * This dynamics model is explicitly used in Q-learning, for better exploration
# * The DQN part inherits from DQN_dist_svgd_agent


import torch.nn.functional as F
from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *
import torchvision
import torch.autograd as autograd
import sys
import tqdm


def gen_ensemble_tensor(tensor, n_particles):
    ones_mask = torch.ones(tensor.dim()).long().tolist()
    tensor = tensor.unsqueeze(0).repeat(
        n_particles, *ones_mask)  # z.shape[1] = particles
    if tensor.dim() > 2 and tensor.size(2) == 1:  # DM lab has incompatible sizing with gym
        tensor = tensor.squeeze(2)
    return tensor


class DynamicsDQNActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()
        self.k = np.random.choice(config.particles, 1)[0]
        self.mdp_update = False
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
            q_max = q_values[abs_max]

        #q_max = to_np(q_max).flatten()
        q_var = q_values.var(0)
        q_mean = q_values.mean(0)
        q_random = to_np(q_values[self.k])

        posterior_z = self._network.sweep_samples()
        posterior_q = self._network(state, seed=posterior_z)

        # random action for exploration and sampled q greedy action for rollout
        if self._total_steps < config.exploration_steps: 
            action = np.random.randint(0, 3)
        else:
            action = np.argmax(q_random)  # Random Head Action

        next_state, reward, done, info = self._task.step([action])
        if config.render and self._task.record_now:
            self._task.render()
        if done:
            self.mdp_update = True
            self.update_steps = self.episode_steps
            self.episode_steps = 0
            self._network.sample_model_seed()
            self.k = np.random.choice(config.particles, 1)[0]
            if self._task.record:
                self._task.record_or_not(info)

        # Add Q value estimates to info
        info[0]['q_mean'] = q_mean.mean()
        info[0]['q_var'] = q_var.mean()
        info[0]['p_var'] = posterior_q.var(0).mean()

        entry = [self._state[0], action, reward[0], 
            next_state[0], int(done[0]), info]
        self._total_steps += 1
        self._state = next_state
        self.episode_steps += 1
        return entry


class Dynamics_Param_Resample_Agent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.config.save_config_to_yaml()
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DynamicsDQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.mdp = config.mdp_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.mdp_optimizer = config.mdp_optimizer_fn(self.mdp.parameters())
        if config.alpha_scheduler == 'linear':
            self.mdp_alpha_schedule = BaselinesLinearSchedule(
                config.alpha_anneal, config.alpha_final, config.alpha_init)
        else:
            self.mdp_alpha_schedule = OneOverSqrtNSchedule(initial_p=config.alpha_init)
        self.actor.set_network(self.network)

        self.total_steps = 0
        self.network.sample_model_seed()
        self.target_network.set_model_seed(self.network.model_seed)
        self.head = np.random.choice(config.particles, 1)[0]
        self.save_net_arch_to_file()
        print(self.network)

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
        # self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state, read_only=True)
        action = self.network.predict_action(state, pred='mean', to_numpy=True)
        action = np.array([action])
        # self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config

        if not torch.equal(self.network.model_seed['value_z'],
                           self.target_network.model_seed['value_z']):
            self.target_network.set_model_seed(self.network.model_seed)

        # rollout env
        transitions = self.actor.step()
        # experiences = []
        pos_experiences = []
        neg_experiences = []
        for state, action, reward, next_state, done, info in transitions:
            self.record_online_return(info)
            self.total_steps += 1
            reward = config.reward_normalizer(reward)

            if reward > 0:
                pos_experiences.append(
                    [state, action, reward, next_state, done])
            else:
                neg_experiences.append(
                    [state, action, reward, next_state, done])

            # experiences.append([state, action, reward, next_state, done])
        # self.replay.feed_batch(experiences)

        if len(pos_experiences) > 0:
            self.replay.feed_pos_batch(pos_experiences)
        if len(neg_experiences) > 0:
            self.replay.feed_neg_batch(neg_experiences)

        if self.total_steps == self.config.exploration_steps:
            print('pure exploration finished')
            self.train_mdp(train_steps=1000)
            self.mdp.sample_model_seed()
            # self.mdp.sample_model_seed(particles=1)

        # models training
        if self.total_steps > self.config.exploration_steps:
            # alpha = self.alpha_schedule.value(self.total_steps)

            ## mdp training
            self.train_mdp()
            self.mdp.sample_model_seed()
            # self.mdp.sample_model_seed(particles=1)
            # if self.actor.mdp_update:
            #     self.train_mdp()
            #     self.actor.mdp_update = False
            #     self.mdp.sample_model_seed()

            ## sample training exp from the replay buffer
            experiences = self.replay.sample_balanced()
            states, actions, rewards, next_states, terminals = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)
            terminals = tensor(terminals).float()
            sample_z = self.network.sample_model_seed(return_seed=True)

            # ## compute max actions for states
            # q_ensemble = self.network(states, seed=sample_z).detach()
            # actions_max = torch.argmax(q_ensemble, dim=-1, keepdim=True) # [p, batch, 1]
            # # print('actions_max var:{}'.format(actions_max.double().var(dim=0).mean()))
            
            # ## form extended actions for q learning
            # ensemble_actions = gen_ensemble_tensor(actions, self.config.particles)
            # ensemble_actions = ensemble_actions.unsqueeze(-1)
            # # [particles, 2 * batch, 1]
            # actions_ext = torch.cat([ensemble_actions, actions_max], dim=1).long()

            ## sample random actions for extended states
            perturb = np.random.randint(
                1, self.config.action_dim, size=actions.shape[0])
            actions_rand = np.remainder(
                actions + perturb, self.config.action_dim)
            actions_rand = tensor(actions_rand).long()
            actions = tensor(actions).long()
            actions_ext = torch.cat([actions, actions_rand], dim=0)
            
            ## repeat states batch to get states_ext
            ones_mask = np.ones(states.ndim - 1, dtype=np.int32).tolist()
            states_ext = np.tile(states, [2, ] + ones_mask)
            
            ## get pred (next_states_rand, rewards_rand) for (states, actions_rand)
            next_states_rand, rewards_rand = self.mdp(
                states, actions_rand) # , seed=sample_z)  # [particles, batch, d_output]
            # next_states_rand = next_states_rand.transpose(0, 1).detach()
            next_states_rand = next_states_rand.detach()
            next_states = tensor(next_states)
            next_states = gen_ensemble_tensor(next_states, self.config.particles)
            next_states_ext = torch.cat([next_states, next_states_rand], dim=1)

            rewards_rand = rewards_rand.squeeze(-1).detach()
            rewards = tensor(rewards)
            rewards = gen_ensemble_tensor(rewards, self.config.particles)
            rewards_ext = torch.cat([rewards, rewards_rand.squeeze()], dim=1)

            ## get target q values
            # [particles, 2*batch, actions]
            q_next = self.target_network(
                next_states_ext, seed=sample_z, ensemble_input=True).detach()
            if self.config.double_q:
                q = self.network(
                    next_states_ext, seed=sample_z, ensemble_input=True)
                best_actions = torch.argmax(
                    q, dim=-1, keepdim=True)  # [particles, 2*batch, 1]
                q_next = q_next.gather(-1, best_actions).squeeze(-1) # [particles, 2*batch]
            else:
                q_next = q_next.max(-1)[0]
            q_target = self.config.discount * \
                q_next * (1 - terminals.repeat(2))
            q_target.add_(rewards_ext)

            ## get main Q values
            q = self.network(states_ext, seed=sample_z) # [particles, 2*batch, 3]
            # q = q.gather(-1, actions_ext) # [particles, 2*batch, 1]
            # q = q.transpose(0, 1) # [2*batch, particles, 1]
            batch_indices = range_tensor(self.replay.batch_size * 2)
            q = q[:, batch_indices, actions_ext]  # [particles, 2*batch]
            q = q.transpose(0, 1).unsqueeze(-1)  # [2*batch, particles, 1]

            q_target = q_target.transpose(
                0, 1).unsqueeze(-1)  # [2*batch, particles, 1]

            ## standard Q-learning
            td_loss = (q_target.detach() - q).pow(2).mul(0.5)
            self.optimizer.zero_grad()
            td_loss.sum().backward()

            for param in self.network.parameters():
                if param.grad is not None:
                    param.grad.data *= 1./config.particles

            if self.config.gradient_clip:
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.gradient_clip)

            with config.lock:
                self.optimizer.step()
            self.logger.add_scalar('td_loss', td_loss.mean(), self.total_steps)

        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def train_mdp(self, alpha=None, train_steps=None):
        config = self.config
        if alpha is None:
            alpha = self.mdp_alpha_schedule.value(self.total_steps)
        if train_steps is None:
            # train_steps = self.actor.update_steps
            train_steps = 1
        for trian_iter in range(train_steps):
            experiences = self.replay.sample_balanced()
            ## experiences = self.replay.sample_balanced()
            states, actions, rewards, next_states, _ = experiences
            states = self.config.state_normalizer(states, read_only=True)
            next_states = self.config.state_normalizer(
                next_states, read_only=True)
            states = tensor(states)
            next_states = tensor(next_states)
            if states.dim() == 3 and states.size(1) == 1:  # DM lab has incompatible sizing with gym
                states = states.squeeze(1)
                next_states = next_states.squeeze(1)
            actions = tensor(actions)
            next_states = next_states.unsqueeze(0).repeat(
                config.particles, 1, 1)
            rewards = tensor(rewards).unsqueeze(-1)
            rewards = rewards.unsqueeze(0).repeat(config.particles, 1, 1)
            sample_z = self.mdp.sample_model_seed(return_seed=True)
            self.mdp.generate_theta(sample_z)  # get params for prediction
            reward_theta_j = self.mdp.reward_theta
            mdp_theta_j = self.mdp.mdp_theta

            ## get pred next states and rewards
            pred_states, pred_rewards = self.mdp(
                states, actions) # [particles, batch, d_out]

            """ Point of consideration # 1 splitting the gradients """
            reward_loss = (rewards.detach() - pred_rewards).pow(2).mul(0.5)
            reward_grad = autograd.grad(reward_loss.sum(), inputs=reward_theta_j)[0]
            
            # [particles/2, batch, d_state]
            #pred_states_i, pred_states_j = torch.split(
            #    pred_states, config.particles//2, dim=0)
            #pred_rewards_i, pred_rewards_j = torch.split(
            #    pred_rewards, config.particles//2, dim=0)
            #target_states_i, target_states_j = torch.split(
            #    next_states, config.particles//2, dim=0)
            #target_rewards_i, target_rewards_j = torch.split(
            #    rewards, config.particles//2, dim=0)

            ## reward function svgd 
            # reward_loss = (target_rewards_j.detach() - pred_rewards_j).pow(2).mul(0.5)
            # [particles/2, batch, 1]
            # reward_grad = autograd.grad(reward_loss.sum(), inputs=pred_rewards_j)[0]

            #pred_rewards_i_eps = pred_rewards_i + torch.rand_like(pred_rewards_i) * 1e-8
            #pred_rewards_j_eps = pred_rewards_j + torch.rand_like(pred_rewards_j) * 1e-8
            # reward_theta_i, reward_theta_j = torch.split(reward_theta, config.particles//2, dim=0)

            
            # kappa: [p/2, p/2], grad_kappa: [p/2, p/2, batch, 1]
            #kappa_reward, grad_kappa_reward = batch_rbf_xy(pred_rewards_j_eps, pred_rewards_i_eps)
            sample_z = self.mdp.sample_model_seed(return_seed=True)
            self.mdp.generate_theta(sample_z)  # get params for prediction
            reward_theta_i = self.mdp.reward_theta
            mdp_theta_i = self.mdp.mdp_theta
            
            reward_theta_j = reward_theta_j.unsqueeze(1)
            reward_theta_i = reward_theta_i.unsqueeze(1)
            kappa_reward, grad_kappa_reward = batch_rbf_mean(reward_theta_j, reward_theta_i)

            # [particles/2, batch, 1]
            reward_grad = reward_grad.unsqueeze(1)   # [p/2, 1, theta]
            p_ref = kappa_reward.shape[0]
            kernel_reward_grad = torch.einsum('ij,ikl->jkl', kappa_reward, reward_grad) / p_ref
            svgd_reward = (kernel_reward_grad + alpha * grad_kappa_reward.mean(0))

            self.mdp_optimizer.zero_grad()
            autograd.backward(reward_theta_i, grad_tensors=svgd_reward.detach(), retain_graph=True)

            ## transition function svgd 
            mdp_loss = F.mse_loss(pred_states, states.detach(), reduction='none')
            # [particles/2, batch, d_state]
            mdp_grad = autograd.grad(mdp_loss.sum(), inputs=mdp_theta_j)[0]
            #mdp_grad_i, mdp_grad_j = torch.split(mdp_grad, config.particles//2, dim=0)
            #mdp_grad = autograd.grad(mdp_loss.sum(), inputs=pred_states_j)[0]

            # kappa: [p/2, p/2], grad_kappa: [p/2, p/2, batch, d_state]
            #mdp_theta_i, mdp_theta_j = torch.split(mdp_theta, config.particles//2, dim=0)
            mdp_theta_j = mdp_theta_j.unsqueeze(1)
            mdp_theta_i = mdp_theta_i.unsqueeze(1)
            kappa_mdp, grad_kappa_mdp = batch_rbf_mean(mdp_theta_j, mdp_theta_i)
            #kappa_mdp, grad_kappa_mdp = batch_rbf_xy(pred_states_j, pred_states_i)

            # [particles/2, batch, d_state]
            mdp_grad = mdp_grad.unsqueeze(1) # [p/2, 1, theta]
            kernel_mdp_grad = torch.einsum('ij,ikl->jkl', kappa_mdp, mdp_grad) / p_ref
            svgd_mdp = (kernel_mdp_grad + alpha * grad_kappa_mdp.mean(0))

            autograd.backward(mdp_theta_i, grad_tensors=svgd_mdp.detach())

            for param in self.mdp.parameters():
                if param.grad is not None:
                    param.grad.data *= 1./config.particles

            if config.gradient_clip:
                nn.utils.clip_grad_norm_(
                    self.mdp.parameters(), config.gradient_clip)

            with config.lock:
                self.mdp_optimizer.step()

            self.logger.add_scalar(
                'reward_loss', reward_loss.mean(), self.total_steps)
            self.logger.add_scalar('reward_kappa', kappa_reward.mean(), self.total_steps)
            self.logger.add_scalar(
                'reward_grad_kappa', grad_kappa_reward.mean(), self.total_steps)

            self.logger.add_scalar(
                'mdp_loss', mdp_loss.mean(), self.total_steps)
            self.logger.add_scalar('mdp_kappa', kappa_mdp.mean(), self.total_steps)
            self.logger.add_scalar(
                'mdp_grad_kappa', grad_kappa_mdp.mean(), self.total_steps)
