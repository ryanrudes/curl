"""
Runs an experiment to see if the stable-baselines3 implementation
of SAC can solve the reacher task
"""

from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC

from encoder import PixelEncoder
    
import utils

import torch.nn as nn
import torch

import numpy as np
import dmc2gym
import wandb
import gym

class VideoRecorder(gym.Wrapper):
    def __init__(self, env, record_freq):
        super().__init__(env)
        self.record_freq = record_freq
        self.episode = 0
        self.timesteps = 0
        self._max_episode_steps = 1000
        
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        
        self.start_timesteps = self.timesteps
        self.record_this_episode = self.timesteps % self.record_freq == 0
        
        if self.record_this_episode:
            self.frames = [observation]
            self.episode_returns = 0
            
        return observation
    
    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        self.timesteps += 1
        
        if self.record_this_episode:
            self.frames.append(observation)
            self.episode_returns += reward
            
            if terminal:
                self.episode += 1
                self.frames = np.array(self.frames)
                self.episode_returns = int(self.episode_returns)
                video = wandb.Video(self.frames, caption = f"t={self.start_timesteps}, R={self.episode_returns}", fps = 60, format = "mp4")
                wandb.log({"train/rollouts": video})
            
        return observation, reward, terminal, info

def view_as_windows(image, shape):
    windows = image.unfold(2, shape[0], 1)
    return windows.unfold(1, shape[1], 1)
    
def random_crop(imgs, output_size):
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = imgs.permute(0, 2, 3, 1)
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (output_size, output_size))
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs

class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=50):
        super().__init__(observation_space, features_dim)
        self.cnn = PixelEncoder((9, 84, 84), features_dim, 4, 32, True)

    def forward(self, observations):
        if observations.shape[0] == 1:
            observations = utils.center_crop_image(observations[0], 84).unsqueeze(0)
        else:
            observations = random_crop(observations, 84)
        return self.cnn(observations)

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 1000000,
    "domain": "reacher",
    "task": "hard",
}

run = wandb.init(
    entity = "ryan-steven",
    project = "CURL",
    config = config,
    sync_tensorboard = True,
    monitor_gym = True,
    save_code = True,
)

env = dmc2gym.make(
    domain_name = "reacher",
    task_name = "hard",
    visualize_reward = False,
    from_pixels = True,
    height = 100,
    width = 100,
    frame_skip = 8,
)
env = Monitor(env)
env = VideoRecorder(env, 10000)
env = utils.FrameStack(env, k = 3)

policy_kwargs = dict(
    features_extractor_class=CNNFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=50),
    net_arch=dict(
        pi=[1024, 1024],
        qf=[1024, 1024],
    )
)

model = SAC("CnnPolicy", env,
    learning_rate = 1e-3,
    buffer_size = 100000,
    learning_starts = 1000,
    batch_size = 128,
    tau = 0.01,
    gamma = 0.99,
    train_freq = 2,
    ent_coef = "auto_0.1",
    target_update_interval = 1,
    target_entropy = -env.action_space.shape[0],
    verbose = 2,
    tensorboard_log = f"runs/{run.id}",
    stats_window_size = 1,
    policy_kwargs = policy_kwargs,
)

model.learn(
    total_timesteps = 1000000,
    callback = WandbCallback(
        gradient_save_freq = 100,
        model_save_path = f"models/{run.id}",
        verbose = 2,
    ),
)