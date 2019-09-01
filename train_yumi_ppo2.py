import sys, os

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from stable_baselines import PPO2
from yumi import *
import argparse
from scipy import stats

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--render', help='render', default=False, action='store_true')
parser.add_argument('--xml-path', help='path to model xml', default='models/cheezit_0.xml')

args = parser.parse_args()

log_dir = "/tmp/yumi/ppo2/{}".format(int(time.time()))
os.makedirs(log_dir, exist_ok=True)

path = args.xml_path
name = os.path.basename(path)
print('name: ', name)

n_cpu = 32

ranges = [(0.15, 0.3), (-0.45, 0.045)]
ranges = [(0.23, 0.23), (0.035, 0.035)]


def make_env(path, render, i, seed=0):
    def create_yumi():
        dynamics = lambda: [np.random.uniform(lo, hi) for lo, hi in ranges]
        return YuMi(path, render=render, seed=seed, dynamics=dynamics)

    return create_yumi

seeds = np.arange(n_cpu)
env = SubprocVecEnv([make_env(path, args.render, i, seed=i) for i in range(n_cpu)])

n_steps = 0
total_timesteps = int(100e6)

def callback(_locals, _globals):
    global n_steps
     
    n_steps += 1
    if n_steps % 100 == 0:
        print('Saving: ', n_steps)
        model.save('checkpoints/yumi/ppo2/ppo2-{}-{}'.format(name, n_steps))

    return True


model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=total_timesteps, callback=callback)
model.save("ppo-yumi-{}-final".format(n_steps))

env.save_running_average(log_dir)

obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
