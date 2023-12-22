import math
import time
import sim
import numpy as np
from numpy import array
from autonomy.RLheighthold import RealSBlimpEnv, DDPG, gym



class Robot():
    
    def __init__(self, frame_name, motor_names=[], client_id=0):  
        # If there is an existing connection
        if client_id:
            self.client_id = client_id
        else:
            self.client_id = self.open_connection()
            
        self.motors = self._get_handlers(motor_names) 
        
        # Robot frame
        self.frame =  self._get_handler(frame_name)
            
        
    def open_connection(self):
        sim.simxFinish(-1)  # just in case, close all opened connections
        self.client_id = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to CoppeliaSim 
        
        if self.client_id != -1:
            print('Robot connected')
        else:
            print('Connection failed')
        return self.client_id
        
    def close_connection(self):    
        sim.simxGetPingTime(self.client_id)  # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive.
        sim.simxFinish(self.client_id)  # Now close the connection to CoppeliaSim:
        print('Connection closed')
    
    def isConnected(self):
        c,result = sim.simxGetPingTime(self.client_id)
        # Return true if the robot is connected
        return result > 0         
        
    def _get_handler(self, name):
        err_code, handler = sim.simxGetObjectHandle(self.client_id, name, sim.simx_opmode_blocking)
        return handler
    
    def _get_handlers(self, names):
        handlers = []
        for name in names:
            handler = self._get_handler(name)
            handlers.append(handler)
        
        return handlers

    def send_motor_velocities(self, vels):
        for motor, vel in zip(self.motors, vels):
            err_code = sim.simxSetJointTargetVelocity(self.client_id, 
                                                      motor, vel, sim.simx_opmode_streaming)      
            
    def set_position(self, position, relative_object=-1):
        if relative_object != -1:
            relative_object = self._get_handler(relative_object)        
        sim.simxSetObjectPosition(self.client_id, self.frame, relative_object, position, sim.simx_opmode_oneshot)

    def simtime(self):
        return sim.simxGetLastCmdTime(self.client_id)
    
    def get_position(self, relative_object=-1):
        # Get position relative to an object, -1 for global frame
        if relative_object != -1:
            relative_object = self._get_handler(relative_object)
        res, position = sim.simxGetObjectPosition(self.client_id, self.frame, relative_object, sim.simx_opmode_blocking)        
        return np.array(position)
    
    def get_velocity(self, relative_object=-1):
        # Get velocity relative to an object, -1 for global frame
        if relative_object != -1:
            relative_object = self._get_handler(relative_object)
        res, velocity, omega = sim.simxGetObjectVelocity(self.client_id, self.frame, sim.simx_opmode_blocking)        
        return array(velocity), array(omega)
    
    
    
    def get_object_position(self, object_name):
        # Get Object position in the world frame
        err_code, object_h = sim.simxGetObjectHandle(self.client_id, object_name, sim.simx_opmode_blocking)
        res, position = sim.simxGetObjectPosition(self.client_id, object_h, -1, sim.simx_opmode_blocking)
        return array(position)
    
    def get_object_relative_position(self, object_name):        
        # Get Object position in the robot frame
        err_code, object_h = sim.simxGetObjectHandle(self.client_id, object_name, sim.simx_opmode_blocking)
        res, position = sim.simxGetObjectPosition(self.client_id, object_h, self.frame, sim.simx_opmode_blocking)
        return array(position)
    
    def set_float(self, f, signal='f'):
        return sim.simxSetFloatSignal(self.client_id, signal, f, sim.simx_opmode_oneshot_wait)

    def set_servo_forces(self, servo_angle1, servo_angle2, force_motor1, force_motor2):
        self.set_float(force_motor1, 'f1')  # Force motor 1
        self.set_float(force_motor2, 'f2')  # Force motor 2
        self.set_float(servo_angle1, 't1')  # Servo 1
        self.set_float(servo_angle2, 't2')  # Servo 2

    def set_sblimp_command(self, fx, fy, tz, servo_angle):
        self.set_float(fx, 'fx')  # Force motor 1
        self.set_float(fy, 'fy')  # Force motor 2
        self.set_float(tz, 'tz')  # Servo 1
        self.set_float(servo_angle, 'servo')  # Servo 2


class SimSBlimpEnv(RealSBlimpEnv):
    def __init__(self, max_steps=100):
        self.goal_height = 4.0
        self.height = 0.0
        self.robot = Robot('bicopterBody')
        self.num_steps = 0
        self.max_steps = max_steps
        self.action_space = gym.spaces.box.Box(np.array([-2, -1]), np.array([2, 1]),
                                               shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.box.Box(np.array([-1]),
                                                    np.array([7]),
                                                    shape=(1,), dtype=np.float32)

    def step(self, action):
        # an action is composed of fx, tau_z, and the servo angle
        # Let's ignore fy for now
        punish = -0.1
        fz, tz = action
        fx = abs(fz)
        servo = np.sign(fz)

        print(action)
        self.robot.set_sblimp_command(fx, 0, tz, (servo * np.pi/2 + np.pi/2))

        self.num_steps += 1

        # finds if state is terminal, and its reward
        reward, terminated = self.evaluate()
        return self._get_obs(), reward + punish, terminated, False, self._get_info()

    def reset(self, seed=None, options=None):
        # print("reset!!!")
        self.num_steps = 0
        self.robot.set_sblimp_command(0, 0, 0, np.pi/2)

        # time.sleep(1)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _get_obs(self):
        feedback = self.robot.get_position()  # get sensor data from robot
        self.height = feedback[2]
        return np.array([self.height], dtype=np.float32)

    def _get_info(self):
        return {}

    def update_state(self):
        feedback = self.robot.get_position()  # get sensor data from robot
        self.height = feedback[2]

    
sim_sblimp_env = SimSBlimpEnv()
#establishes the model using DDPG with a multi layer perceptron policy/critic
model = DDPG("MlpPolicy",  sim_sblimp_env, verbose=1, learning_starts=50, gamma=0.99, device="cuda")
model.learn(total_timesteps=100000, log_interval=4)
model.save("ddpg_sblimp1")
