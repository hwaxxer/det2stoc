import os
import sys
import time
import numpy as np
import mujoco_py
from mujoco_py import load_model_from_path, MjSim, MjViewer, functions
from collections import deque
import xml.etree.ElementTree as EE
from gym.spaces import Box
from enum import Enum
import imageio
from scipy import stats

np.set_printoptions(precision=4, suppress=True)
MAX_TIME = 4.0


class WindySlope(object):

    def __init__(self, model, mode, hertz=25, should_render=True, should_screenshot=False):
        self.hertz = hertz
        self.steps = 0
        self.should_render = should_render
        self.should_screenshot = should_screenshot
        self.nsubsteps = int(MAX_TIME / model.opt.timestep / 100)
        self.viewer = None
        self.model = model
        self.mode = mode

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.sim = MjSim(model)
        self.data = self.sim.data
 
        if self.should_render:
            self.viewer = MjViewer(self.sim)
            self.viewer.cam.azimuth = 45
            self.viewer.cam.elevation = -20
            self.viewer.cam.distance = 25
            self.viewer.cam.lookat[:] = [0, 0, -2]
            self.viewer.scn.flags[3] = 0
            self.viewer._hide_overlay = True

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
        self.viewer._hide_overlay = True
        self.viewer.render()
        width, height = 2560, 1440
        img = self.viewer.read_pixels(width, height, depth=False)
        # original image is upside-down, so flip it
        img = img[::-1, :, :]
        imageio.imwrite(image_path, img)

    def step(self):
        nsubsteps = self.nsubsteps
        for _ in range(nsubsteps):
            self.sim.step()
            self.render()
        self.steps += 1

        return self.get_observations(self.model, self.data)

    def render(self):
        if self.should_render:
            self.viewer.render()


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

    mode = Mode.NORMAL

    friction = 0.23
    insidebox = 0.11
    wind = -1.7
    friction_scale = 0.1
    insidebox_scale = 0.2
    wind_scale = 1.0

    n_episodes = 100 if mode == Mode.REAL else 10000

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
    snapshot_step = 99

    for ep in range(n_episodes):

        f, i, w = sample_parameters(mode)
        path = os.path.join(os.getcwd(), './windyslope.xml')
        model = load_model_from_path(path)
        model = randomize_dynamics(model, friction=f, insidebox=i, wind=w)
        env = WindySlope(model, mode, should_render=should_render)

        obs = env.reset()

        screenshot_step = 100
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

            obs = env.step()

            l.append(obs)
            l.append(np.array([f, i, w]))
            l.append(step)
            l = np.array(l)
            observations.append(l)


            if True and mode == mode.REAL and 0 <= snapshot_step and step+1 == snapshot_step:
                replay(time, snapshot_step, qpos, qvel, qacc)
                snapshot_step = -1

            if mode == mode.REAL and step+1 in [1, 50, 100]:
                env.screenshot('traj/windyslope-real-all-obs-{}.png'.format(step+1))
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

