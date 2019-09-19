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
uniform = True

DYNAMICS_LO = [0.1, -0.085]
DYNAMICS_HI = [1, 0.085]

if real:
    params = [0.35, 0.08]
else:
    params = [(0.1, 0.9), (-0.09, 0.09)]
    #params = [(0.3248, 0.1152), (0.0788, 0.0284)]
    #params = [(0.69048023, 0.3243859), (0.12308636, 0.04219555)]
    #params = [(0.82035035,0.108042516), (0.09504627, 0.021840865)]
    #params = [(0.14548126, 0.46216267), (0.03220001, 0.057417072)]

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
        return YuMi(path=path, task=args.task,
                render=render, seed=seed,
                dynamics=dynamics_generator(seed),
                logging_level=logging_level)

    return create_yumi

logging_level = logging.DEBUG if args.debug else logging.INFO
yumis = [make_env(args.render, i, seed=i) for i in range(n_cpu)]
env = SubprocVecEnv(yumis)

model = PPO2.load(args.model_path, env=env, policy=MlpPolicy)

n_episodes = 100 if real else 5000

states, actions, next_states, parameters, steps = [], [], [], [], []

horizon = env.env_method('get_horizon')[0]
n_steps = (horizon*n_episodes)//n_cpu

deterministic = False

obs = env.reset()
for ep in range(n_steps):
    states.extend(obs)
    action, _states = model.predict(obs, deterministic=deterministic)
    obs, rewards, done, info = env.step(action)
    actions.extend(action)
    next_states.extend(obs)
    dynamics = env.env_method('get_dynamics')
    parameters.extend(dynamics)
    steps.append(env.env_method('get_step'))

data = {
        'states': np.array(states),
        'actions': np.array(actions),
        'next_states': np.array(next_states),
        'parameters': np.array(parameters),
        'steps': np.array(steps)
        }

mode_str = 'real' if real else 'fake'
dist_str = 'uniform' if uniform else 'normal'
basename = os.path.basename(args.model_path)
filename = 'new_data/{}/yumi/{}_yumi_ppo_{}_fri_{}_com{}_{}_det{}.npz'.format(basename,
        mode_str, n_episodes, str(params[0]), str(params[1]), dist_str,
        int(deterministic))

os.makedirs(os.path.dirname(filename), exist_ok=True)

with open(filename, 'wb') as f:
    np.save(f, data)
