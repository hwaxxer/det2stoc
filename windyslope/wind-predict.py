import os
import sys
import time
import numpy as np
import mujoco_py
from mujoco_py import load_model_from_path, MjSim, MjViewer, functions
from collections import deque
import xml.etree.ElementTree as EE
import gym
from gym.spaces import Box
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym.wrappers import Monitor
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv
from enum import Enum
import imageio
from scipy import stats

np.set_printoptions(precision=4, suppress=True)
MAX_TIME = 4.0

def softplus_inverse(x):
    return tf.log(tf.expm1(x))

def normalize(inp, activation, reuse, scope, norm='None'):
    if norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp


class WindySlope(gym.Env):

    def __init__(self, model, mode, hertz=25, should_render=True, should_screenshot=False):
        self.hertz = hertz
        self.steps = 0
        self.should_render = should_render
        self.should_screenshot = should_screenshot
        self.nsubsteps = int(MAX_TIME / model.opt.timestep / 100)
        self.viewer = None
        self.model = model
        self.mode = mode
        self.enabled = True
        self.metadata = {'render.modes': 'rgb_array'}
        self.should_record = True

    def close(self):
        pass

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(18,))

    @property
    def action_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(0,))

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.sim = MjSim(model)
        self.data = self.sim.data
 
        if self.should_render:
            if self.viewer:
                self.viewer.sim = sim
                return
            self.viewer = MjViewer(self.sim)
            self.viewer.cam.azimuth = 45
            self.viewer.cam.elevation = -20
            self.viewer.cam.distance = 25
            self.viewer.cam.lookat[:] = [0, 0, -2]
            self.viewer.scn.flags[3] = 0

    def reset(self):
        self.sim.reset()
        self.steps = 0

        self.sim.forward()

        obs = self.get_observations(self.model, self.data)
        return obs

    def get_observations(self, model, data):
        self.sim.forward()
        obs = []
        name = 'box'
        pos = data.get_body_xpos(name)
        xmat = data.get_body_xmat(name).reshape(-1)
        velp = data.get_body_xvelp(name)
        velr = data.get_body_xvelr(name)

        for x in [pos, xmat, velp, velr]:
            obs.extend(x.copy())

        obs = np.array(obs, dtype=np.float32)
        return obs

    def screenshot(self, image_path):
        self.viewer.hide_overlay = True
        self.viewer.render()
        width, height = 2560, 1440
        #width, height = 1,1
        img = self.viewer.read_pixels(width, height, depth=False)
        # original image is upside-down, so flip it
        img = img[::-1, :, :]
        imageio.imwrite(image_path, img)

    def step(self, action):
        nsubsteps = self.nsubsteps
        for _ in range(nsubsteps):
            self.sim.step()
            self.render()
        self.steps += 1

        return self.get_observations(self.model, self.data), 1, self.steps == 100, {}

    def render(self, mode=None):
        if self.should_render:
            self.viewer._overlay.clear()
            self.viewer.render()
            if self.should_record:
                def nothing():
                    return
                self.viewer._create_full_overlay = nothing
                wind = self.model.opt.wind[0]
                self.viewer.add_overlay(1, "Wind", "{:.2f}".format(wind))
                width, height = 2560, 1440
                img = self.viewer.read_pixels(width, height, depth=False)
                # original image is upside-down, so flip it
                img = img[::-1, :, :]
                return img

    def euler2quat(self, euler):
        euler = np.asarray(euler, dtype=np.float64)
        assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

        ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
        si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
        ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
        cc, cs = ci * ck, ci * sk
        sc, ss = si * ck, si * sk

        quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
        quat[..., 0] = cj * cc + sj * ss
        quat[..., 3] = cj * sc - sj * cs
        quat[..., 2] = -(cj * ss + sj * cc)
        quat[..., 1] = cj * cs - sj * sc
        return quat

    def degrees2radians(self, degrees):
        return degrees * np.pi / 180

class Mode(Enum):
    REAL = 0
    UNIFORM = 1
    NORMAL = 2

def truncnorm(lo, hi, scale, mean=None):
    if mean is None:
        mean = (hi-lo)/2+lo
    return stats.truncnorm((lo-mean)/scale, (hi-mean)/scale, loc=mean, scale=scale)

def randomize_dynamics(model, friction, insidebox, wind):
    pair_friction = model.pair_friction
    pair_friction = pair_friction[0]
    pair_friction[:2] = friction

    model.opt.wind[0] = wind

    bodyid = model.body_name2id('insidebox')
    model.body_pos[bodyid][0] = insidebox
    return model

def replay(time, step, qpos, qvel, qacc, n=1000):
    for ep in range(n):
        f, i, w = sample_parameters(mode)
        randomize_dynamics(model, friction=f, insidebox=i, wind=w)
        env = WindySlope(model, mode.REAL, should_render=should_render)
        env.data.time = time
        env.data.qpos[:] = qpos
        env.data.qvel[:] = qvel
        env.data.qacc_warmstart[:] = qacc
        assert env.sim.get_state() == sim_state

        obs_before = env.get_observations(env.model, env.data)
        assert np.allclose(qpos, env.data.qpos[:]), 'nopes'
        l = []
        l.append(obs_before.copy())
        obs_after = env.step()
        l.append(obs_after.copy())
        l.append(np.array([f, i, w]))
        l.append(step)
        l = np.array(l)
        states.append(l)

        assert np.count_nonzero(before - obs_before) == 0, 'wrong'

    filename = 'states-step{}.npz'.format(step)
    with open(filename, 'wb') as f:
        np.save(f, states)

def replay_states(path):
    f, i, w = sample_parameters(mode)
    path = os.path.join(os.getcwd(), './windyslope.xml')
    model = load_model_from_path(path)
    randomize_dynamics(model, friction=f, insidebox=i, wind=w)
    env = WindySlope(model, mode.REAL, should_render=should_render)
    env.reset()
    with open('traj100-sample10.npz', 'rb') as f:
        states = np.load(f)

    if env.should_record:
        rec = VideoRecorder(env, path='/tmp/video/windyslope-predict-sample.mp4')

    for e in range(len(states)):
        episode = states[e]
        for i in range(len(episode)):
            qpos = episode[i][:3]
            env.data.qpos[:3] = qpos
            mat = episode[i][3:12]
            mat = np.asarray(mat).astype(np.float64)
            quat = np.empty(4)
            functions.mju_mat2Quat(quat, mat)
            print('quat: ', quat)
            env.data.qpos[3:] = quat
            #env.data.qvel[:3] = states[i][12:15]
            #env.data.qvel[3:] = states[i][15:18]
            
            #env.sim.forward()
            obs = env.get_observations(env.model, env.data)
            print('states:', episode[i])
            print('obs:', obs)
            #assert np.allclose(obs, states[i])
            env.render()
            if env.should_record:
                rec.capture_frame()
    

def sample_parameters(mode):
    if mode == Mode.UNIFORM:
        f = truncnorm(friction_lo, friction_hi, scale=friction_scale).rvs()
        i = truncnorm(insidebox_lo, insidebox_hi, scale=insidebox_scale).rvs()
        w = truncnorm(wind_lo, wind_hi, scale=wind_scale).rvs()
    elif mode == Mode.NORMAL:
        f = np.random.normal(friction_mean, friction_std)
        i = np.random.normal(insidebox_mean, insidebox_std)
        w = np.random.normal(wind_mean, wind_std)
    else:
        f = np.random.normal(friction, 0.01)
        i = np.random.normal(insidebox, 0.04)
        w = np.random.normal(wind, 0.05)

    return f, i, w

if __name__ == '__main__':


    np.random.seed(6)

    should_render = False
    if len(sys.argv) > 1:
        should_render = sys.argv[1].lower() == 'true'

    mode = Mode.UNIFORM

    friction = 0.23
    insidebox = 0.11
    wind = -1.7
    friction_scale = 0.1
    insidebox_scale = 0.2
    wind_scale = 1.0

    n_episodes = 1 if mode == Mode.REAL else 10

    if mode == Mode.UNIFORM:
        friction_lo, friction_hi = 0.1, 0.4
        insidebox_lo, insidebox_hi = -0.25, 0.25
        wind_lo, wind_hi = -3, 1
        name = ('data/obstacle/windyplane_{}_f-lo{:.2f}-hi{:.2f}_i-lo{:.4f}-hi{:.4f}_wind-lo{:.2f}-hi{:.2f}'
            .format(n_episodes, friction_lo, friction_hi, insidebox_lo, insidebox_hi, wind_lo, wind_hi))
    elif mode == Mode.NORMAL:
        name = ('data/obstacle/windyplane_{}_friction-mean{:.4f}-std{:.4f}_insidebox-mean{:.4f}-std{:.4f}_wind-mean{:.2f}-std{:.4f}'
            .format(n_episodes, friction_mean, friction_std, insidebox_mean, insidebox_std, wind_mean, wind_std))
    else:
        name = ('data/obstacle/real_windyplane_{}_friction{}_insidebox{}_wind{}'
            .format(n_episodes, friction, insidebox, wind))


    observations = []
    states = []
    snapshot_step = -1 

    np.random.seed(4)

    replay_states('traj10.npz')
    raise 'OK'

    def make_env():
        env = WindySlope(model, mode, should_render=should_render)
        return env
    f, i, w = sample_parameters(mode)
    path = os.path.join(os.getcwd(), './windyslope.xml')
    model = load_model_from_path(path)
    env = make_env()
    if env.should_record:
        rec = VideoRecorder(env, path='/tmp/video/windyslope.mp4')

    wind = np.linspace(-3, 1, n_episodes)
    for ep in range(n_episodes):

        f, i, w = sample_parameters(mode)
        w = wind[ep]
        path = os.path.join(os.getcwd(), './windyslope.xml')
        #model = load_model_from_path(path)
        model = randomize_dynamics(model, friction=f, insidebox=i, wind=w)
        #env.model = model

        obs = env.reset()

        screenshot_step = 1000
        if should_render:
            env.render()
            env.screenshot('master/windyslope/traj/windyslope-real-all-obstacle-{}.png'.format(screenshot_step))

        print('f: {}, i: {}, wind: {}'.format(f, i, w))

        for step in range(100):
            if mode == Mode.REAL and 0 <= snapshot_step and step+1 == snapshot_step:
                qpos = env.data.qpos.copy()
                qvel = env.data.qvel.copy()
                qacc = env.data.qacc.copy()
                time = env.data.time
                sim_state = env.sim.get_state()
                before = env.get_observations(env.model, env.data)

            l = []
            feature = {}

            l.append(obs)

            if env.should_record:
                rec.capture_frame()
            obs = env.step([None])

            l.append(obs)
            l.append(np.array([f, i, w]))
            l.append(step)
            l = np.array(l)
            observations.append(l)


            if mode == mode.REAL and 0 <= snapshot_step and step+1 == snapshot_step:
                replay(time, snapshot_step, qpos, qvel, qacc)
                snapshot_step = -1

            if step+1 == screenshot_step:
                geomid = env.model.geom_name2id('box')
                pos = env.data.geom_xpos[geomid]
                mat = env.data.geom_xmat[geomid].reshape(-1)
                quat = np.empty(4)
                functions.mju_mat2Quat(quat, np.asarray(mat))

                geomid = env.model.geom_name2id('insidebox')
                hpos = env.data.geom_xpos[geomid]
                hmat = env.data.geom_xmat[geomid].reshape(-1)
                hquat = np.empty(4)
                functions.mju_mat2Quat(hquat, np.asarray(hmat))
                states.append(np.hstack([pos.copy(), quat.copy(), hpos.copy(), hquat.copy()]))

    with open('states-{}.npz'.format(screenshot_step), 'wb') as f:
        np.save(f, states)

    with open(name + '.npz', 'wb') as f:
        np.save(f, observations)

    if env.should_record:
        rec.close()
