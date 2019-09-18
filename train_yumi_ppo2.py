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
parser.add_argument('--checkpoint-path', help='path to previous checkpoint', type=str, default=None)
parser.add_argument('--xml-path', help='path to model xml', default='models/cheezit.xml')
parser.add_argument('--task', help='task to solve', default=0, type=int)

args = parser.parse_args()

log_dir = "/tmp/yumi/ppo2/{}".format(int(time.time()))
os.makedirs(log_dir, exist_ok=True)

path = args.xml_path
name = os.path.basename(path)

n_cpu = 6

params  = [0.23, 0.085]

def make_env(path, render, i, seed=0):
    def create_yumi():
        dynamics = lambda: params
        return YuMi(path, task=args.task, render=render, seed=seed, dynamics=dynamics)

    return create_yumi

seeds = np.arange(n_cpu)
env = SubprocVecEnv([make_env(path, args.render, i, seed=i) for i in range(n_cpu)])

n_steps = 0
finetune = args.checkpoint_path is None
total_timesteps = int(100e6)

def callback(_locals, _globals):
    global n_steps
     
    n_steps += 1
    if n_steps % 10 == 0 or n_steps < 10:
        if finetune:
            save_path = os.path.join(log_dir, 'ppo2_finetune_{}_task_{}_{}.npy'.format(name, args.task, n_steps))
        else:
            save_path = os.path.join(log_dir, 'ppo2_{}_task_{}_{}.npy'.format(name, args.task, n_steps))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)

    return True

if args.checkpoint_path is None:
    model = PPO2(MlpPolicy, env, n_steps=1024, verbose=1, tensorboard_log=log_dir)
else:
    model = PPO2.load(args.checkpoint_path, env=env, policy=MlpPolicy, tensorboard_log=log_dir)

model.learn(total_timesteps=total_timesteps, callback=callback)

obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
