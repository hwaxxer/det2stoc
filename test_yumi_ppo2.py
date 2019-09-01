import gym
import sys
import argparse

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

from scipy import stats

from yumi import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model_path', help='name of model')
parser.add_argument('--xml-path', help='path to model xml', default='./models/cheezit_0.xml')
parser.add_argument('--render', help='render', default=False, action='store_true')

args = parser.parse_args()

path = args.xml_path

n_cpu = 1

real = True
if real:
    params = [(0.23, 0.006), (0.035, 0.005)]
else:
    params = [(0.1, 0.4), (-0.049, 0.049)]

    means = [0.2963142, 0.03320432]
    stds = [0.047035545, 0.010614053]

    # small

    means = [0.28041244, 0.035019487]
    stds = [0.050948095, 0.010011874]

    means = [0.30144903, 0.036558267]
    stds = [0.045239843, 0.003935931]

    params = [(m,s) for m,s in zip(means, stds)]


def dynamics_params(seed):
    if real:
        return [np.random.normal(m,s) for m,s in params]
    else:
        return [np.random.normal(lo,hi) for lo,hi in params]

def dynamics_generator(seed):
    return lambda: dynamics_params(seed)

def make_env(render, i, seed=0):
    def create_yumi():
        return YuMi(path=path, render=render, seed=seed, dynamics=dynamics_generator(seed))

    return create_yumi

yumis = [make_env(args.render, i, seed=i) for i in range(n_cpu)]
env = SubprocVecEnv(yumis)
#env = DummyVecEnv(yumis)

model = PPO2.load(args.model_path, env=env, policy=MlpPolicy)

n_episodes = 300 if real else 100
observations = []

obs = env.reset()
for ep in range(n_episodes):
    for step in range(100):
        l = [[] for _ in range(n_cpu)]
        for i in range(n_cpu):
            l[i].append(obs[i])
        action, _states = model.predict(obs, deterministic=False)
        dynamics = env.env_method('get_dynamics')
        #if ep == 0 and step == 1:
        #    dynamics = env.env_method('screenshot')
        obs, rewards, done, info = env.step(action)
        for i in range(n_cpu):
            l[i].extend([action[i], obs[i]])
            l[i].append(dynamics[i])
        observations.extend([np.array(x) for x in l])

if real:
    filename = 'data/yumi/real_yumi_ppo_{}_friction{}_com{}.npz'.format(n_episodes, str(params[0]), str(params[1]))
else:
    filename = 'data/yumi/fake_yumi_ppo_{}_friction_{}_com{}.npz'.format(n_episodes, str(params[0]), str(params[1]))

with open(filename, 'wb') as f:
    np.save(f, observations)

