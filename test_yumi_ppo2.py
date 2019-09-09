import gym
import sys
import argparse

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

from scipy import stats

from yumi import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model_path', help='name of model')
parser.add_argument('--xml-path', help='path to model xml', default='./models/cheezit.xml')
parser.add_argument('--task', help='the task to perform', default=0, type=int)
parser.add_argument('--render', help='render or not', default=False, action='store_true')
parser.add_argument('--debug', help='debug or not', default=False, action='store_true')
parser.add_argument('--n_cpu', help='how many processes to run', default=1, type=int)

args = parser.parse_args()
path = args.xml_path
n_cpu = args.n_cpu

assert not args.render or (args.render and 1 == n_cpu), 'Cannot render when using more than one cpu.'

real = True
if real:
    params = [0.25, -0.08]
else:
    params = [(0.0, 0.5), (-0.085, 0.085)]

def dynamics_params(seed):
    if real:
        return params
    else:
        return [np.random.uniform(lo,hi) for lo,hi in params]

def dynamics_generator(seed):
    return lambda: dynamics_params(seed)

def make_env(render, i, seed=0):
    def create_yumi():
        logging_level = logging.DEBUG if args.debug else logging.INFO
        return YuMi(path=path, task=args.task, render=render, seed=seed, dynamics=dynamics_generator(seed), logging_level=logging_level)

    return create_yumi

yumis = [make_env(args.render, i, seed=i) for i in range(n_cpu)]
env = SubprocVecEnv(yumis)

model = PPO2.load(args.model_path, env=env, policy=MlpPolicy)

n_episodes = 100 if real else 5000
n_episodes //= n_cpu

observations = []

horizon = env.env_method('get_horizon')[0]

obs = env.reset()
for ep in range(n_episodes):
    dynamics = env.env_method('get_dynamics')
    print('Dynamics: ', dynamics)
    for step in range(horizon):
        l = [[] for _ in range(n_cpu)]
        for i in range(n_cpu):
            l[i].append(obs[i])
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, done, info = env.step(action)
        for i in range(n_cpu):
            l[i].extend([action[i], obs[i]])
            l[i].append(dynamics[i])
        observations.extend([np.array(x) for x in l])

mode_str = 'real' if real else 'fake'
filename = 'data/yumi/{}_yumi_ppo_{}_mass_{}_com{}.npz'.format(mode_str, n_episodes, str(params[0]), str(params[1]))

with open(filename, 'wb') as f:
    np.save(f, observations)

