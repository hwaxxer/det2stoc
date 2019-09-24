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
import logging
from gym.envs.robotics import rotations
from pid import PID


np.set_printoptions(precision=4, suppress=True)
MAX_TIME = 4.0
VELOCITY_CONTROLLER = False

class YuMi(gym.GoalEnv):

    metadata = {'render.modes': ['human']}

    def __init__(self, path, dynamics, task, goal_env=False, hertz=25, render=True, seed=0, logging_level=logging.INFO):
        logging.basicConfig(level=logging_level)
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.goal_env = goal_env
        self.task = task

        self.path = path
        model = load_model_from_path(path)

        self.joint_idx = []
        for name in sorted(model.joint_names):
            if 'yumi' not in name:
                continue
            self.joint_idx.append(model.joint_name2id(name))

        self.joint_states_pos = None
        self.joint_states_vel = None
        self.previous_action = np.zeros(14)
        self.target_hist = deque(maxlen=2)
        self.hertz = hertz
        self.steps = 0
        self.should_render = render
        self.steps_per_action = int(1.0 / hertz / model.opt.timestep)
        model.nuserdata = 14
        self.viewer = None
        self.model = model

        self._set_joint_limits()

        self.generate_dynamics = dynamics

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.sim = MjSim(model, udd_callback=self.callback)
        self.data = self.sim.data
        if self.should_render and not self.viewer:
            self.viewer = MjViewer(self.sim)
            self.viewer.cam.azimuth = 180
            self.viewer.cam.elevation = -15
            self.viewer._hide_overlay = True

    @property
    def horizon(self):
        return int(self.hertz * MAX_TIME)

    def get_horizon(self):
        return self.horizon

    @property
    def action_space(self):
        return spaces.Box(low=-0.1, high=0.1, shape=(14,), dtype=np.float32)

    @property
    def observation_space(self):
        d = spaces.Dict({
            'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(91,), dtype=np.float32),
            'desired_goal': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            'achieved_goal': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
            })
        if self.goal_env:
            return d
        else:
            return d['observation']

    def get_step(self):
        return self.steps

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
            terminal = True

        self.render()
        return terminal

    def callback(self, state):
        if self.joint_states_pos is None:
            return

        model = state.model
        nu = model.nu
        data = state.data

        dt = 1/self.hertz

        if VELOCITY_CONTROLLER:
            action = np.empty(nu)
            action[:] = data.userdata[:nu]
            ctrl = action - self.joint_states_vel
            data.qacc[self.joint_idx] = ctrl * 1.0 / dt
        else:
            action = np.empty(model.nu)
            for i in range(len(self.joint_idx)):
                pid = self.pids[i]
                qpos = self.joint_states_pos[i]
                action[i] = pid(qpos, data.time)
            res = action
            res -= self.joint_states_vel * dt
            data.qacc[self.joint_idx] = (1.0 / dt**2) * res

        functions.mj_inverse(model, data)
        data.qfrc_applied[self.joint_idx] = data.qfrc_inverse[self.joint_idx]


    def reset(self):

        self.steps = 0
        self.joint_states_pos = None
        self.joint_states_vel = None
        self.data.qacc[:] = 0
        self.data.qvel[:] = 0

        self.pids = []
        for i in range(self.model.nu):
            pid = PID(Kp=1.0, Kd=0.001, sample_time=0)
            pid.output_limits = (self.action_space.low[i], self.action_space.high[i])
            self.pids.append(pid)

        _ = self.randomize_dynamics(self.model)
        self.sim.reset()

        self.set_initial_pose()
        self.sim.forward()

        obs = self.get_observations()
        return obs

    def set_initial_pose(self):
        self.data.set_joint_qpos('yumi_joint_1_l', 1.)
        self.data.set_joint_qpos('yumi_joint_1_r', -1.)
        self.data.set_joint_qpos('yumi_joint_2_l', 0.1)
        self.data.set_joint_qpos('yumi_joint_2_r', 0.1)

        ''' randomize goal
        goal_id = self.model.body_name2id('goal')
        pos = self.model.body_pos[goal_id]
        pos[0:2] = [0.5, 0]
        pos[0:2] += np.random.uniform(-0.05, 0.05, size=(2,))
        self.model.body_pos[goal_id] = pos
        '''

        self.sim.forward()
        target_start, target_end = self.model.get_joint_qpos_addr('target')
        target_qpos = self.data.qpos[target_start:target_end]

        target_quat = target_qpos[3:]

        if self.task == 0:
            # rotate y=-90 deg
            quat = rotations.euler2quat(np.array([0, -np.pi/2, 0]))
            z_idx = 0
        elif self.task == 1:
            # rotate x=90 deg
            quat = rotations.euler2quat(np.array([np.pi/2, 0, 0]))
            z_idx = 1
        elif self.task == 2:
            # do nothing
            quat = rotations.euler2quat(np.array([0, 0, 0]))
            z_idx = 2
        elif self.task == 3:
            # rotate z=-90 deg 
            quat = rotations.euler2quat(np.array([0, 0, -np.pi/2]))
            z_idx = 2
        else:
            raise Exception('Additional tasks not implemented.')

        target_id = self.model.geom_name2id('target')
        target_qpos[0] = 0.5 + np.random.uniform(-0.01, 0.01)
        target_qpos[1] = np.random.uniform(0.14, 0.15)
        height = self.model.geom_size[target_id][z_idx]
        target_qpos[2] = 0.051 + height

        goal_id = self.model.body_name2id('goal')
        body_pos = self.model.body_pos[goal_id]
        body_quat = self.model.body_quat[goal_id]
        body_pos[2] = target_qpos[2]
        body_quat[:] = quat

        perturbation = np.zeros(3)
        perturbation[z_idx] = np.random.uniform(-0.2, 0.2)
        euler = rotations.quat2euler(quat)
        euler = rotations.subtract_euler(euler, perturbation)
        target_qpos[3:] = rotations.euler2quat(euler)

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
            name = model.joint_names[joint]
            assert model.get_joint_qpos_addr(name) == joint
        obs.extend(pos)
        obs.extend(vel)

        self.joint_states_pos = np.array(pos) + np.random.normal(0, 0.001, size=len(pos))
        self.joint_states_vel = np.array(vel) + np.random.normal(0, 0.001, size=len(vel))

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

        observations['desired_goal'] = self.get_desired_goal()
        observations['achieved_goal'] = self.get_achieved_goal()

        while len(self.target_hist) < self.target_hist.maxlen:
            self.target_hist.append(np.hstack([pos, mat]))
        self.target_hist.append(np.hstack([pos, mat]))

        obs.extend(self.target_hist[-1])
        obs.extend((self.target_hist[-1] - self.target_hist[-2])/(1/self.hertz))

        target_id = model.geom_name2id('target')
        size = model.geom_size[target_id]
        obs.extend(size)

        obs = np.array(obs)
        observations['observation'] = obs
        if self.goal_env:
            return observations
        else:
            return obs

    def render(self):
        if self.should_render:
            if False and self.sim.nsubsteps == self.steps_per_action and self.steps % 10 != 0:
                return

            self.viewer.render()

    def step(self, action):
        action = .1*action

        self.steps += 1

        reward = 0
        terminal = False
        # Check for early termination
        terminal, force_penalty = self.bad_collision()
        if terminal:
            reward = -10

        # Eventhough limits are specified in action_space, they 
        # are not honored by baselines so we clip them
        action = np.clip(action, self.action_space.low, self.action_space.high)

        idx = 0
        if self.joint_states_pos[idx] > 1.2:
            action[0] = min(action[idx], 0)
        elif self.joint_states_pos[0] < 0.8:
            action[0] = max(action[0], 0)

        idx = 1
        if self.joint_states_pos[idx] < -1.2:
            action[idx] = max(action[idx], 0)
        elif self.joint_states_pos[idx] > -0.8:
            action[idx] = min(action[idx], 0)

        for idx in [2,3,4,5]:
            if self.joint_states_pos[idx] > 0.4:
                action[idx] = min(action[idx], 0)
            if self.joint_states_pos[idx] < -0.3:
                action[idx] = max(action[idx], 0)

        idx = -2
        if self.joint_states_pos[idx] > 0.2:
            action[idx] = min(action[idx], 0)
        elif self.joint_states_pos[idx] < -0.4:
            action[idx] = max(action[idx], 0)

        idx = -1
        if self.joint_states_pos[idx] < -0.2:
            action[idx] = max(action[idx], 0)
        elif self.joint_states_pos[idx] > 0.4:
            action[idx] = min(action[idx], 0)

        if VELOCITY_CONTROLLER:
            action += self.joint_states_vel
        else:
            action += self.joint_states_pos

        for i,a in enumerate(action):
            pid = self.pids[i]
            pid.setpoint = a

        # Perturbate action
        #action += np.random.normal(0, 0.001, size=len(action))

        stop_early = self.simstep()
        terminal = terminal or stop_early
        obs = self.get_observations()

        if not terminal:
            reward = self.compute_reward(self.get_achieved_goal(), self.get_desired_goal(), {})
            #reward -= force_penalty
            #logger.debug('force penalty: %f' % force_penalty)
            terminal = self.steps == self.horizon

        if self.steps % self.hertz == 0:
            logging.info('Step: {}, reward: {}'.format(
                self.steps, reward))
            if 0.8 < reward and self.steps == self.horizon:
                logging.info('**** LOOKING GOOD ****')

        return obs, reward, terminal, {}

    def get_distance(self, a, b):
        return np.linalg.norm(a - b, axis=-1)

    def get_goal_distance(self, achieved_goal, desired_goal):
        pos1, pos2 = achieved_goal[:], desired_goal[:]
        pos_distance = self.get_distance(pos1, pos2)
        return pos_distance

    def compute_reward(self, achieved_goal, desired_goal, info):
        pos_distance = self.get_goal_distance(achieved_goal, desired_goal)
        pos_reward = self.get_pos_reward(pos_distance)

        euler1, euler2 = achieved_goal[3:], desired_goal[3:]
        ang_distance = np.linalg.norm(rotations.subtract_euler(euler1, euler2), axis=-1)
        ang_distance_ratio = 0.5
        ang_distance_penalty = ang_distance_ratio*ang_distance

        reward = pos_reward - ang_distance_penalty

        if self.steps % 10 == 0:
            logging.debug('Reward: %f, pos reward: %f, ang penalty: %f' % (reward, pos_reward, ang_distance_penalty))
        return reward

    def get_pos_reward(self, distance, close=0.01, margin=0.2):
        return max(0, 1-distance/margin)

    def _set_joint_limits(self):
        xml_str = self.model.get_xml()
        tree = EE.fromstring(xml_str)
        low, high = [], []
        for name in sorted(self.model.joint_names):
            if 'yumi' not in name:
                continue
            limit = tree.find(".//joint[@name='{}']".format(name))
            limit_range = [float(x) for x in limit.get('range').split(' ')]
            low.append(limit_range[0])
            high.append(limit_range[1])
        self.joint_limits = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)

    def bad_collision(self):
        bad_collision = False
        force_penalty = 0

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            bodyid1 = self.model.geom_bodyid[geom1]
            bodyid2 = self.model.geom_bodyid[geom2]

            bodyname1 = self.model.body_id2name(bodyid1)
            bodyname2 = self.model.body_id2name(bodyid2)

            is_target = 'target' in bodyname1 or 'target' in bodyname2
            is_target = is_target or 'box-composite' in bodyname1 or 'box-composite' in bodyname2
            is_table = 'table' in bodyname1 or 'table' in bodyname2

            if is_target and is_table:
                continue
            elif is_target:
                continue
                sim = self.sim
                body1_cfrc = sim.data.cfrc_ext[bodyid1]
                body1_contact_force_norm = np.sqrt(np.sum(np.square(body1_cfrc)))
                body2_cfrc = sim.data.cfrc_ext[bodyid2]
                body2_contact_force_norm = np.sqrt(np.sum(np.square(body2_cfrc)))

                force_penalty = body1_contact_force_norm + body2_contact_force_norm
            else:
                bad_collision = True
                if bad_collision:
                    print('body is: ', bodyname1, bodyname2)
                    break

        return bad_collision, force_penalty

    def get_dynamics(self):
        return self.dynamics

    def randomize_dynamics(self, model):
        self.dynamics = self.generate_dynamics()
        fri, icom = self.dynamics

        try:
            id = model.body_name2id('insidebox')
            model.body_pos[id][-1] = icom

            for pair in range(model.npair):
                tableid = model.geom_name2id('table')
                targetid = model.geom_name2id('target')
                if ((model.pair_geom1 == tableid and model.pair_geom2 == targetid) or
                   (model.pair_geom2 == tableid and model.pair_geom1 == targetid)):

                    pair_friction = model.pair_friction[pair]
                    pair_friction[:2] = [fri, fri]

            logging.debug('Dynamics: {}'.format(self.dynamics))
        except:
            pass

        self.sim.forward()
        return model

    def get_desired_goal(self):
        return self.get_site_pose('site:goal')

    def get_achieved_goal(self):
        return self.get_site_pose('site:target')

    def get_site_pose(self, site):
        xpos = self.data.get_site_xpos(site)
        euler = rotations.mat2euler(self.data.get_site_xmat(site))
        return np.hstack([xpos, euler])

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
