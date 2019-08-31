import os
import random
import time
import numpy as np
import mujoco_py
from mujoco_py import load_model_from_path, MjSim, MjViewer, functions
from collections import deque
import xml.etree.ElementTree as EE
import gym
from gym import spaces
import tensorflow as tf


np.set_printoptions(precision=7, suppress=True)
MAX_TIME = 4.0 
STEPS_PER_RENDER = 10
VELOCITY_CONTROLLER = True

class YuMi(gym.GoalEnv):

    metadata = {'render.modes': ['human']}

    def __init__(self, dynamics, hertz=25, render=True, seed=0):

        print('seed: ', seed)
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        path = os.path.join(os.getcwd(), './models/cheezit_0.xml')
        self.path = path
        model = load_model_from_path(path)

        self.joint_idx = []
        for name in sorted(model.joint_names):
            if name == 'target':
                continue
            print('Name: ', name)
            self.joint_idx.append(model.joint_name2id(name))

        print('joint idx: ', self.joint_idx)
        self.joint_states_pos = None
        self.joint_states_vel = None
        self.target_hist = deque(maxlen=2)
        self.hertz = hertz
        self.steps = 0
        self.should_render = render
        self.steps_per_action = int(1.0 / hertz / model.opt.timestep)
        model.nuserdata = 14
        self.viewer = None
        self.model = model

        self._set_joint_limits()

        self.dynamics = dynamics

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.sim = MjSim(model, udd_callback=self.callback)
        #self.sim = MjSim(model, nsubsteps=self.steps_per_action, udd_callback=self.callback)
        self.data = self.sim.data
        if self.should_render and not self.viewer:
            self.viewer = MjViewer(self.sim)
            self.viewer.cam.azimuth = 180
            self.viewer.cam.elevation = -15
            self.viewer._hide_overlay = True

    @property
    def horizon(self):
        return self.hertz * MAX_TIME

    @property
    def action_space(self):
        return spaces.Box(low=-0.01, high=0.01, shape=(14,), dtype=np.float32)

    @property
    def observation_space(self):
        d = spaces.Dict({
            'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(94,), dtype=np.float32),
            'desired_goal': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'achieved_goal': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
            })
        return d

    def simstep(self):
        try:
            if self.sim.nsubsteps == self.steps_per_action:
                terminal = self.sim.step()
            else:
                # TODO: randomize substeps
                for step in range(self.steps_per_action):
                    terminal = self.sim.step()
                    if terminal:
                        break

        except Exception as e:
            print('Caught exception: ', e)
            with open('exception.txt', 'w') as f:
                f.write(str(e))

        self.render()
        return terminal

    def callback(self, state):
        if self.joint_states_vel is None:
            return

        model = state.model
        nu = model.nu
        data = state.data

        dt = model.opt.timestep
        if self.sim.nsubsteps == self.steps_per_action:
            dt *= self.steps_per_action

        dt = 1/self.hertz

        if VELOCITY_CONTROLLER:
            action = np.empty(nu)
            action[:] = data.userdata[:nu]
            ctrl = action - self.joint_states_vel
            data.qacc[self.joint_idx] = ctrl * 1.0 / dt
        else:
            action = np.empty(model.nu)
            action[:] = data.userdata[:nu]
            res = action
            res -= data.qvel[:nu] * dt
            data.qacc[:nu] = (2.0 / dt**2) * res
            #data.qacc[:nu] = np.clip(data.qacc[:nu], -10., 10.)

        functions.mj_inverse(model, data)
        data.qfrc_applied[self.joint_idx] = data.qfrc_inverse[self.joint_idx]
        

    def reset(self):

        self.steps = 0
        self.joint_states_pos = None
        self.joint_states_vel = None
        self.data.qacc[:] = 0
        self.data.qvel[:] = 0

        model = load_model_from_path(self.path)
        self.randomize_dynamics(model)
        #self.model = model
        self.sim.reset()

        self.set_initial_pose()

        self.sim.forward()

        obs = self.get_observations()
        #print('Reset returning: ', obs)
        return obs

    def set_initial_pose(self):
        self.data.set_joint_qpos('yumi_joint_1_l', 1.)
        self.data.set_joint_qpos('yumi_joint_1_r', -1.)
        self.data.set_joint_qpos('yumi_joint_2_l', 0.2)
        self.data.set_joint_qpos('yumi_joint_2_r', 0.2)

        ''' randomize goal
        goal_id = self.model.body_name2id('goal')
        pos = self.model.body_pos[goal_id]
        pos[0:2] = [0.5, 0]
        pos[0:2] += np.random.uniform(-0.05, 0.05, size=(2,))
        self.model.body_pos[goal_id] = pos
        '''

        start, end = self.model.get_joint_qpos_addr('target')
        qpos = self.data.qpos[start:end]
        qpos[0:2] += np.random.uniform(-0.02, 0.02, size=(2,))

    def quat2mat(self, quat):
        result = np.empty(9, dtype=np.double)
        functions.mju_quat2Mat(result, np.asarray(quat))
        return result

    def mat2quat(self, mat):
        result = np.empty(4, dtype=np.double)
        functions.mju_mat2Quat(result, np.asarray(mat))
        return result

    def get_observations(self):
        """ 
        This functions looks a bit awkward since it is trying to replicate
        the order of observations when listening to transforms using ROS TF.
        """
        obs = []
        observations = {}

        pos = []
        vel = []
        qacc = []

        model = self.model
        data = self.data

        # Previous policy in ROS used translations between yumi_base_link
        # and requested frames.
        # We subtract the xpos of yumi_base_link to mimic that behavior.
        base_link_xpos = data.get_body_xpos('yumi_base_link')

        # TODO Look at this
        for i, joint in enumerate(self.joint_idx):
            pos.append(data.qpos[joint])
            vel.append(data.qvel[joint])
            qacc.append(data.qacc[joint])
            name = model.joint_names[joint]
            assert model.get_joint_qpos_addr(name) == joint
        obs.extend(pos)
        obs.extend(vel)

        self.joint_states_pos = np.array(pos)
        self.joint_states_vel = np.array(vel)
        self.qacc = qacc

        name = 'site:goal'
        pos = data.get_site_xpos(name)
        pos -= base_link_xpos
        mat = data.get_site_xmat(name)
        mat = mat.reshape(-1)
        obs.extend(pos)
        obs.extend(mat)

        names = ['left_gripper_base', 'right_gripper_base']
        for name in names:
            pos = data.get_body_xpos(name)
            pos -= base_link_xpos
            quat = data.get_body_xquat(name)
            mat = self.quat2mat(quat)
            obs.extend(pos)
            obs.extend(mat)

        name = 'site:target'
        pos = data.get_site_xpos(name)
        pos -= base_link_xpos
        mat = data.get_site_xmat(name)
        mat = mat.reshape(-1)

        desired_goal = self.data.get_site_xpos('site:goal')
        achieved_goal = self.data.get_site_xpos('site:target')
        observations['desired_goal'] = desired_goal 
        observations['achieved_goal'] = achieved_goal

        while len(self.target_hist) < self.target_hist.maxlen:
            self.target_hist.append(np.hstack([pos, mat]))
        self.target_hist.append(np.hstack([pos, mat]))
        obs.extend(np.ravel(self.target_hist))

        target_id = model.geom_name2id('target')
        mass = model.body_mass[target_id]
        obs.append(mass)
        size = model.geom_size[target_id]
        obs.extend(size)

        obs.extend(self.get_dynamics())

        obs = np.array(obs)
        observations['observation'] = obs
        return observations

    def render(self):
        if self.should_render:
            if False and self.sim.nsubsteps == self.steps_per_action and self.steps % 10 != 0:
                return

            self.viewer.render()

    def step(self, action):

        action_penalty = 0.001*np.linalg.norm(action)
        # Eventhough limits are specified in action_space, they 
        # are not honored by baselines so we clip them

        action = np.clip(action, self.action_space.low, self.action_space.high)
        action += self.joint_states_vel

        # MAX_TIME seconds
        self.steps += 1

        completed = terminal = False

        # Check for early termination
        terminal = self.bad_collision()

        obs = self.get_observations()

        if self.joint_states_pos[2] > 0.3:
            action[2] = min(action[2], 0)
        if self.joint_states_pos[3] > 0.3:
            action[3] = min(action[3], 0)

        if not terminal:
            self.data.userdata[:] = action

            stop_early = self.simstep()

            achieved_goal, desired_goal = obs['achieved_goal'], obs['desired_goal']
            reward = self.compute_reward(achieved_goal, desired_goal, {})

            completed = reward == 1.0
            reward -= action_penalty
            terminal = self.steps == self.horizon

            if self.steps % self.hertz == 0:
                print('Step: {}, reward: {}, completed: {}'.format( 
                    self.steps, reward, completed))
                if completed:
                    print('************ Completed ************')
        else: 
            reward = -3

        return obs, reward, terminal, {}

    def goal_distance(self, a, b):
        return np.linalg.norm(a - b, axis=-1)

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = self.goal_distance(achieved_goal, desired_goal)
        reward, _ = self.get_reward(distance)
        return reward

    def get_reward(self, distance, close=0.035, margin=0.15):
        # sigmoidal gaussian
        reward = 0 
        completed = distance < close
        
        if distance < close:
            reward = 1.0
        elif distance < margin:
            scaled_distance = distance / margin;
            scale = np.sqrt(-2 * np.log(margin))
            reward += np.exp(-0.5 * (scale*scaled_distance)**2)
        return reward, completed

    def _set_joint_limits(self):
        xml_str = self.model.get_xml()
        tree = EE.fromstring(xml_str)
        low, high = [], []
        for name in sorted(self.model.joint_names):
            if name == 'target':
                continue
            limit = tree.find(".//joint[@name='{}']".format(name))
            limit_range = [float(x) for x in limit.get('range').split(' ')]
            low.append(limit_range[0])
            high.append(limit_range[1])
        self.joint_limits = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)

    def bad_collision(self):
        bad_collision = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            bodyid1 = self.model.geom_bodyid[geom1]
            bodyid2 = self.model.geom_bodyid[geom2]

            bodyname1 = self.model.body_id2name(bodyid1)
            bodyname2 = self.model.body_id2name(bodyid2)

            rootid1 = self.model.body_rootid[bodyid1]
            rootid2 = self.model.body_rootid[bodyid2]

            root_bodyname1 = self.model.body_id2name(rootid1)
            root_bodyname2 = self.model.body_id2name(rootid2)

            is_yumi = 'yumi' in root_bodyname1 or 'yumi' in root_bodyname2
            is_table = 'table' in bodyname1 or 'table' in bodyname2
            is_target= 'target' in bodyname1 or 'target' in bodyname2

            bad_collision = is_yumi and is_table
            if bad_collision:
                print('root body is: ', root_bodyname1, root_bodyname2)
                print('body is: ', bodyname1, bodyname2)
                break

            if is_table and is_target:
                # Not interested in this contact force
                continue

            sim = self.sim

            body1_cfrc = sim.data.cfrc_ext[bodyid1]
            body1_contact_force_norm = np.sqrt(np.sum(np.square(body1_cfrc)))
            body2_cfrc = sim.data.cfrc_ext[bodyid2]
            body2_contact_force_norm = np.sqrt(np.sum(np.square(body2_cfrc)))

            max_cfrc = 10 # Arbitrary limit 
            if max_cfrc < body1_contact_force_norm or max_cfrc < body2_contact_force_norm:
                bad_collision = True
                print('Contact force on: ', bodyname1, ': ', body1_contact_force_norm)
                print('Contact force on: ', bodyname2, ': ', body2_contact_force_norm)
                break

        return bad_collision

    def get_dynamics(self):
        return [self.friction, self.com]

    def randomize_dynamics(self, model):
        dynamics = self.dynamics()
        print('dyns: ', dynamics)
        friction, com = dynamics

        self.friction = friction
        self.com = com 

        #pair_friction = model.pair_friction[0]
        #pair_friction[:2] = self.friction

        #bodyid = model.body_name2id('insidebox')
        #model.body_pos[bodyid][0] = com

        return dynamics

    def screenshot(self, image_path='image.png'):
        import imageio
        self.viewer._hide_overlay = True
        self.viewer.render()
        width, height = 2560, 1440
        img = self.viewer.read_pixels(width, height, depth=False)
        # original image is upside-down, so flip it
        img = img[::-1, :, :]
        imageio.imwrite(image_path, img)
        #self.viewer._hide_overlay = False