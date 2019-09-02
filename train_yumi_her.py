import sys, os

from stable_baselines import HER, DQN, SAC, DDPG, TD3
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from yumi import *
import argparse
from scipy import stats

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--render', help='render', default=False, action='store_true')
parser.add_argument('--xml-path', help='path to model xml', default='models/cheezit.xml')
parser.add_argument('--task', help='task to solve', default=0, type=int)

args = parser.parse_args()

log_dir = "/tmp/yumi/her/{}".format(int(time.time()))
os.makedirs(log_dir, exist_ok=True)

ranges = [(0.23, 0.23), (0.035, 0.035)]

name = os.path.basename(args.xml_path)

def make_env(render, seed=0):
    def create_yumi():
        dynamics = lambda: [np.random.uniform(lo, hi) for lo, hi in ranges]
        return YuMi(args.xml_path, task=args.task, goal_env=True, render=render, seed=seed, dynamics=dynamics)

    return create_yumi

env = DummyVecEnv([make_env(args.render, seed=17)])

n_steps = 0
total_timesteps = int(100e6)

def callback(_locals, _globals):
    global n_steps
     
    n_steps += 1
    if n_steps % 10000 == 0:
        print('Saving: ', n_steps)
        model.save('checkpoints/yumi/her/her_{}_task_{}_{}.npy'.format(name, args.task, n_steps))

    return True


model = HER('MlpPolicy', env, model_class=DDPG, verbose=1, tensorboard_log=log_dir, **dict(random_exploration=.2))
model.learn(total_timesteps=total_timesteps, callback=callback)
model.save("her-yumi-{}-final".format(n_steps))

env.save_running_average(log_dir)

obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
