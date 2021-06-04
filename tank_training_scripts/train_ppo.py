#!/usr/bin/env python3

import os
import gym
import tankgym
import argparse

from stable_baselines.ppo1 import PPO1
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback

NUM_TIMESTEPS = int(2e7)
SEED = 721
EVAL_SEED = 938
EVAL_FREQ = 250000
EVAL_EPISODES = 500

parser = argparse.ArgumentParser(description='Options for PPO training')
parser.add_argument('--logdir', help='Director of logging the ', type=str, default="ppo_default_log")
parser.add_argument('--batchnum', help='batchnum ', type=int, default=4096)
parser.add_argument('--gamma', help='gamma ', type=float, default=0.99)
parser.add_argument('--optim_stepsize', help='optim_stepsize ', type=float, default=3e-4)
args = parser.parse_args()

LOGDIR = args.logdir
timesteps_per_actorbatch = args.batchnum
optim_stepsize = args.optim_stepsize
gamma = args.gamma

print("***********")
print("Logging to " + LOGDIR)

logger.configure(folder=LOGDIR)

train_env = gym.make("TankGymTrain-v0")
train_env.seed(SEED)
train_env.policy = tankgym.BaselineRandWAim()

eval_env = gym.make("TankGym-v0")
eval_env.seed(EVAL_SEED)
eval_env.policy = tankgym.BaselineRandWAim()

# take mujoco hyperparams (but 2x timesteps_per_actorbatch to cover more steps.)
model = PPO1(MlpPolicy, train_env, timesteps_per_actorbatch=timesteps_per_actorbatch, clip_param=0.2, entcoeff=0.0, optim_epochs=10,
                 optim_stepsize=optim_stepsize, optim_batchsize=64, gamma=gamma, lam=0.95, schedule='linear', verbose=2)

eval_callback = EvalCallback(eval_env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.

train_env.close()
eval_env.close()
