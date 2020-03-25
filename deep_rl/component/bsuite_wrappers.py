import gym
import bsuite

class DeepSeaReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.episode_rewards = 0
        self.total_rewards = 0
        self.ep = 0
        self.ep_steps = 0
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode_rewards += reward
        self.total_rewards += reward
        self.ep_steps += 1
        env_attrs = self.env.bsuite_info()
        total_bad_episodes = env_attrs['total_bad_episodes']
        denoised_return = env_attrs['denoised_return']
        if done:
            self.ep += 1
            info['episodic_return'] = self.episode_rewards
            info['episode'] = self.ep
            info['denoised_return'] = denoised_return
            info['ep_steps'] = self.ep_steps
            info['total_return'] = self.total_rewards
            info['total_bad_episode'] = total_bad_episodes
            self.episode_rewards = 0
            self.ep_steps = 0
        else:
            info['episodic_return'] = None
            info['denoised_return'] = denoised_return
            info['episode'] = self.ep
            info['ep_steps'] = self.ep_steps
            info['total_return'] = self.total_rewards
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


class CartpoleSwingupReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.episode_rewards = 0
        self.total_rewards = 0
        self.ep = 0
        self.ep_steps = 0
        self.bad_episode = False
        self.total_bad_episodes = 0
        self.denoised_return = 0
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode_rewards += reward
        self.total_rewards += reward
        self.ep_steps += 1
        if reward > 0.0005:
            self.upright += 1
            self.total_upright += 1
        if done:
            self.ep += 1
            info['episodic_return'] = self.episode_rewards
            info['episode'] = self.ep
            info['bad_episode'] = self.upright
            info['ep_steps'] = self.ep_steps
            info['total_return'] = self.total_rewards
            info['total_upright'] = self.total_upright
            self.episode_rewards = 0
            self.ep_steps = 0
            self.upright = 0
        else:
            info['episodic_return'] = None
            info['episodic_upright'] = None
            info['episode'] = self.ep
            info['episodic_upright'] = self.upright
            info['ep_steps'] = self.ep_steps
            info['total_return'] = self.total_rewards
            info['total_upright'] = None
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


