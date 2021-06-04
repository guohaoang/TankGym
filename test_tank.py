"""
Testing file for TankGym. Runs AGENT_POLICY against OPP_POLICY for 500 games.

Prints out statistics of the games, including number of wins and losses.
"""

import math
import numpy as np
import gym
import tankgym

np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

# game settings:

RENDER_MODE = False
AGENT_POLICY = tankgym.BaselineRandWAim()
OPP_POLICY = tankgym.BaselineRandWAim()


if __name__=="__main__":

  if RENDER_MODE:
    from pyglet.window import key
    from time import sleep

  policy =  AGENT_POLICY

  env = gym.make("TankGym-v0")
  env.seed(np.random.randint(0, 10000))

  env.policy = OPP_POLICY

  if RENDER_MODE:
    env.render()

  obs = env.reset()

  total_reward = 0

  done = False
  numgames = 500
  rewards = []
  for game in range(numgames):
      env.reset()
      done = False
      while not done:

        action = policy.predict(obs)

        obs, reward, done, _ = env.step(action)

        if reward > 0 or reward < 0:
          manualMode = False
          otherManualMode = False

        total_reward += reward

        if done:
            rewards += [reward]

        if RENDER_MODE:
          env.render()
          sleep(0.02)

  env.close()
  print("cumulative score", total_reward)

  print("games:", policy.getName(), "vs", env.policy.getName())
  print("mean:", np.mean(rewards))
  print("variance:", np.var(rewards))
  print("std:", np.std(rewards))
  print("numwin:", len([x for x in rewards if x == 100]))
  print("numlose:", len([x for x in rewards if x == -100]))
