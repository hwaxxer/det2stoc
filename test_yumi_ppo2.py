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

real = False
uniform = False

DYNAMICS_LO = [-0.085, 0.1]
DYNAMICS_HI = [0.085, 1]

if real:
    params = [0.35, -0.08]
    params = [0.4, 0.08]
else:
    params = [(0.1, 0.5), (-0.085, 0.085)]
    params = [(0.3424, 0.0772), (-0.0521, 0.0153)]

def truncnorm(lo, hi, loc, scale):
    return stats.truncnorm((lo-loc)/scale, (hi-loc)/scale, loc=loc, scale=scale).rvs()

def dynamics_params(seed):
    if real:
        return params
    else:
        if uniform:
            return [np.random.uniform(lo,hi) for lo,hi in params]
        else:
            return [truncnorm(DYNAMICS_LO[i], DYNAMICS_HI[i], loc, scale) for i, (loc,scale) in enumerate(params)]

def dynamics_generator(seed):
    return lambda: dynamics_params(seed)

def make_env(render, i, seed=0):
    def create_yumi():
        logging_level = logging.DEBUG if args.debug else logging.INFO
        return YuMi(path=path, task=args.task, render=render, seed=seed, dynamics=dynamics_generator(seed), logging_level=logging_level)

    return create_yumi

yumis = [make_env(args.render, i, seed=i) for i in range(n_cpu)]
env = SubprocVecEnv(yumis)
env = DummyVecEnv(yumis)

model = PPO2.load(args.model_path, env=env, policy=MlpPolicy)

n_episodes = 300 if real else 1000

horizon = env.env_method('get_horizon')[0]


states, actions, next_states, dynamics = [], [], [], []

'''
obs = env.reset()
for ep in range(n_episodes):
    dynamics = env.env_method('get_dynamics')
    print('Dynamics: ', dynamics)
    for step in range(horizon):
        states.append(obs)
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        actions.append(action)
        next_states.append(obs)
        dynamics.append(dynamics)

data = {'states': np.array(states),
        'actions': np.array(actions),
        'next_states': np.array(next_states),
        'dynamics': np.array(dynamics)}
with open(filename, 'wb') as f:
    np.save(f, data)

'''
mode_str = 'real' if real else 'fake'
dist_str = 'uniform' if uniform else 'normal'
filename = 'data/yumi/low_friction_center_box/{}_yumi_ppo_{}_fri_{}_com{}_deterministic_{}.npz'.format(mode_str, n_episodes, str(params[0]), str(params[1]), dist_str)

observations = []
obs = env.reset()
for ep in range(n_episodes//n_cpu):
    dynamics = env.env_method('get_dynamics')
    print('Dynamics: ', dynamics)
    for step in range(horizon):
        l = [[] for _ in range(n_cpu)]
        for i in range(n_cpu):
            l[i].append(obs[i])
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        for i in range(n_cpu):
            l[i].extend([action[i], obs[i]])
            l[i].append(dynamics[i])
        observations.extend([np.array(x) for x in l])

with open(filename, 'wb') as f:
    np.save(f, observations)

