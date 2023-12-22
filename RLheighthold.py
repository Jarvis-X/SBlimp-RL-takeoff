import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
import time


class RealSBlimpEnv(gym.Env):
    def __init__(self, robot_esp_now, broadcast_channel, slave_index, max_steps=500):
        self.goal_height = 4.0
        self.height = 0.0
        self.robot = robot_esp_now
        self.num_steps = 0
        self.channel = broadcast_channel
        self.index = slave_index
        self.max_steps = max_steps
        self.action_space = gym.spaces.box.Box(np.array([-0.5, -0.1, -1]), np.array([0.5, 0.1, 1]),
                                               shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.box.Box(np.array([-1]),
                                                    np.array([7]),
                                                    shape=(1,), dtype=np.float32)

    def randomize_state_goal(self):
        self.goal_height = 3.0 + np.random.random()*2.0

    def _get_obs(self):
        feedback = self.robot.getFeedback(1)  # get sensor data from robot
        self.height = feedback[0]
        return np.array([self.height], dtype=np.float32)

    def _get_info(self):
        return {}

    def update_state(self):
        feedback = self.robot.getFeedback(1)  # get sensor data from robot
        self.height = feedback[0]

    def reset(self, seed=None, options=None):
        # print("reset!!!")
        self.num_steps = 0
        self.robot.send([21, int(1), 0, 0, 0, 0, 0, 0, int(0 * 90 + 90), 0, 0, 0, 0],
                        self.channel, self.index)  # send control command to robot
        # time.sleep(1)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # an action is composed of fx, tau_z, and the servo angle
        # Let's ignore fy for now
        punish = -0.1
        fx, tz, servo = action

        print(action)
        self.robot.send([21, int(1), fx, 0, 0, 0, 0, tz, int(servo * 90 + 90), 0, 0, 0, 0],
                        self.channel, self.index)  # send control command to robot

        self.num_steps += 1

        # finds if state is terminal, and its reward
        reward, terminated = self.evaluate()
        return self._get_obs(), reward + punish, terminated, False, self._get_info()

    # Evaluates the state for its rewards and terminality
    def evaluate(self):
        terminated = False
        reward = -abs(self.height - self.goal_height)
        if self.num_steps > self.max_steps:
            terminated = True
        return reward, terminated
