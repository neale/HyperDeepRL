######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import os
import gym
import numpy as np
import torch
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from gym.wrappers import Monitor
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.atari_wrappers import FrameStack as FrameStack_
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv

import bsuite
from bsuite.utils import gym_wrapper
from bsuite import bsuite
from ..utils import *
from .bsuite_wrappers import *
try:
    import roboschool
except ImportError:
    pass

import imageio
import matplotlib.pyplot as plt
# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py

class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.episode_rewards = 0
        self.total_rewards = 0
        self.ep = 0
        self.ep_steps = 0
        self.upright = 0
        self.total_upright = 0
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
            info['episodic_upright'] = self.upright
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


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


# The original LayzeFrames doesn't work well
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]


class FrameStack(FrameStack_):
    def __init__(self, env, k):
        FrameStack_.__init__(self, env, k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


# The original one in baselines is really bad
class DummyVecEnv(VecEnv):
    def __init__(self, env_fns, name):
        self.name = name
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        data = []
        for i in range(self.num_envs):
            obs, rew, done, info = self.envs[i].step(self.actions[i])
            if done:
                obs = self.envs[i].reset()
            data.append([obs, rew, done, info])
        obs, rew, done, info = zip(*data)
        return obs, np.asarray(rew), np.asarray(done), info
    
    def reset(self):
        return [env.reset() for env in self.envs]
    
    def get_images(self):
        if 'cartpole' in self.name:
            mode = 'cartpole'
        else:
            mode = 'rgb_array'
        return [env.render(mode=mode) for env in self.envs]
    
    def render(self, mode='human'):
        if 'cartpole' in self.name:
            mode='cartpole'
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def close(self):
        return


class GIFlogger(object):
    def __init__(self, log_dir, record_freq=10, max_len=None):
        self.log_dir = log_dir
        self.record_freq = record_freq
        self.max_gif_len = max_len
        self.stored_frames = []
        self.base_name = 'Episode_'
        self.videos_logged = 1
        self.create_gif_directory()

    def reset_frames(self):
        self.stored_frames = []
    
    def add_frame(self, frame):
        self.stored_frames.append(frame)

    def __len__(self):
        return len(self.stored_frames)
    
    def create_gif_directory(self):
        self.save_dir = self.log_dir+'/episode_gif_recordings/'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def create_and_save(self):
        assert len(self) > 0, "Trying to record GIF with empty buffer"
        fp = self.save_dir+self.base_name+str(self.record_freq*self.videos_logged)+'.gif'
        np_frames = np.array(self.stored_frames)
        imageio.mimsave(fp, np_frames, fps=60)
        self.videos_logged += 1
    

class Task:
    def __init__(self,
                 name,
                 num_envs=1,
                 single_process=True,
                 log_dir=None,
                 episode_life=True,
                 seed=np.random.randint(int(1e5)),
                 video=False,
                 gif=False,
                 video_logging_freq=10,
                 special_args=None):

        self.log_dir = log_dir
        self.video_freq = video_logging_freq
        self.video_enabled = video
        self.gif_enabled = gif
        self.videos_logged = 0
        if self.gif_enabled:
            self.gif_logger = GIFlogger(log_dir, video_logging_freq)
        if video == False and gif == False:
            self.record = False
        else:
            self.record = True

        self.record_now = False
    
        self.name = name
        envs = [self.make_env(name, seed, i, episode_life, special_args) for i in range(num_envs)]
        if single_process:
            Wrapper = DummyVecEnv
        else:
            Wrapper = SubprocVecEnv
        self.env = Wrapper(envs, self.name)
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.env.observation_space.shape))
        if 'Chain' in self.name:
            self.state_dim = self.env.envs[0].n
        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'

    # adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
    def make_env(self, env_id, seed, rank, episode_life=True, special_args=None):
        is_atari = False
        is_bsuite = False
        is_dmcontrol = False

        def _thunk():
            random_seed(seed)
            if env_id.startswith('bsuite'):
                is_bsuite = True
                id = env_id.split('bsuite-')[1]
                self.video_enabled = False
                
                bsuite_env = bsuite.load_and_record_to_csv(
                        bsuite_id=id,
                        results_dir='bsuite_logs/{}'.format(np.random.randint(int(1e6))),
                        overwrite=True
                )
               
                # bsuite_env = bsuite.load_from_id(id)
                env = gym_wrapper.GymFromDMEnv(bsuite_env)
                if 'cartpole_swingup' in id:
                    wrapper = CartpoleSwingupReturnWrapper
                elif 'deep_sea_stochastic' in id:
                    wrapper = DeepSeaStochasticReturnWrapper
                elif 'deep_sea' in id:
                    wrapper = DeepSeaReturnWrapper
                env = wrapper(env)
            elif env_id.startswith("dm"):
                is_dmcontrol = True
                import dm_control2gym
                _, domain, task = env_id.split('-')
                env = dm_control2gym.make(domain_name=domain, task_name=task)
            
            else:
                if special_args is not None:
                    if 'NChain' in special_args[0]:
                        print ('starting chain N = ', special_args[1])
                        env = gym.make(env_id, n=special_args[1])
                else:
                    env = gym.make(env_id)

            if self.video_enabled:
                env = Monitor(env, self.log_dir, video_callable=self.video_callable)

            is_atari = hasattr(gym.envs, 'atari') and isinstance(
                env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
            if is_atari:
                env = make_atari(env_id)
            env.seed(seed + rank)
            if (is_bsuite == False):
                env = OriginalReturnWrapper(env)
            if is_atari:
                env = wrap_deepmind(env,
                                    episode_life=episode_life,
                                    clip_rewards=False,
                                    frame_stack=False,
                                    scale=False)
                obs_shape = env.observation_space.shape
                if len(obs_shape) == 3:
                    env = TransposeImage(env)
                env = FrameStack(env, 4)
            return env

        return _thunk


    def reset(self):
        return self.env.reset()
    
    def record_or_not(self, info):
        if info[0]['episode'] % self.video_freq == 0:
            self.record_now = True
            self.start_record()
        else:
            self.record_now = False
            self.end_record()

    def start_record(self):
        self.gif_logger.reset_frames()

    def end_record(self):
        if len(self.gif_logger) > 0:
            self.gif_logger.create_and_save()
            self.gif_logger.reset_frames()

    def render(self):
        if self.record_now and self.gif_enabled:
            frame = self.env.render()
            self.gif_logger.add_frame(frame)
        return frame

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)


if __name__ == '__main__':
    task = Task('Hopper-v2', 5, single_process=False)
    state = task.reset()
    while True:
        action = np.random.rand(task.observation_space.shape[0])
        next_state, reward, done, _ = task.step(action)
        print(done)
