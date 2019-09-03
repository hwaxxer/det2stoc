import gym
import os
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2


class CartPoleTrainer(object):
    def __init__(self, env):
        self.model = PPO2(MlpPolicy, env, verbose=1)

    def train(self, model_path):
        self.model.learn(total_timesteps=250000)
        self.model.save(model_path)
        return self.model


def generate(parameter_distribution, num_episodes, env_update_fn, filepath=None, n_cpu=6):
    env_name = 'CartPole-v1'
    model_dir = os.path.join(os.getcwd(), 'models')
    model_path = os.path.join(model_dir, 'ppo2_' + env_name + '.pkl')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    def make_env(env_name):
        env = gym.make(env_name)
        return env

    env = SubprocVecEnv([lambda: make_env(env_name) for i in range(n_cpu)])

    try:
        model = PPO2.load(model_path)
    except Exception as e:
        trainer = CartPoleTrainer(env)
        model = trainer.train(model_path)

    obs = env.reset()

    ep_lengths = np.array([0 for _ in range(n_cpu)])

    env = make_env(env_name)

    states, actions, next_states, parameters, steps = [], [], [], [], []

    for ep in range(num_episodes):
        obs = env.reset()
        params = parameter_distribution()
        env_update_fn(env.unwrapped, params)

        done = False
        step = 0
        while not done:
            action, _states = model.predict(obs)
            states.append(obs)
            actions.append([action])
            obs, reward, done, info = env.step(action)
            next_states.append(obs)
            parameters.append(params)
            steps.append(step)


    data = { 
            'states': np.array(states),
            'actions': np.array(actions),
            'next_states': np.array(next_states),
            'parameters': np.array(parameters)
            }
    if filepath:
        print('filepath: ', filepath)
        with open(filepath, 'wb') as f:
            np.save(filepath, data)

    return data

if __name__ == '__main__':

    data_dir = 'data'
    model_dir = 'model'
    for p in [data_dir, model_dir]:
        if not os.path.exists(p):
            os.mkdir(p)

    masspole_dist = lambda: np.random.uniform(0.1, 2.0)
    length_dist = lambda: np.random.uniform(0.5, 4.0)

    parameter_dist = lambda: (masspole_dist(), length_dist())

    num_episodes = 10
    generate(model_dir, parameter_dist, num_episodes)
