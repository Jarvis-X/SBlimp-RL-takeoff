from autonomy.ZigZagWalk import DeterministicWalk
from autonomy.RandomWalk import RandomWalk
from parameters import *
from teleop.joystickHandler import JoystickHandler
from comm.ESPNOW import ESPNOWControl
from robot.robotConfig import RobotConfig
from gui.visualizer import SensorGUI
from autonomy.RLheighthold import RealSBlimpEnv, DDPG

# Override feedback params for multiple robots
YAW_SENSOR = True
Z_SENSOR = True


# User interface




# Joystick
joyhandler = JoystickHandler(yaw_sensor=YAW_SENSOR)

# Communication
esp_now = ESPNOWControl(PORT, LIST_OF_MAC_ADDRESS, ESP_VERBOSE)

# Load robot configuration
robotConfigs = [RobotConfig(esp_now, i, robot_mac) for i, robot_mac in enumerate(LIST_OF_MAC_ADDRESS)]
# GUIS
sensor_guis = [SensorGUI(GUI_ENABLED,robConfig=robConfig) for robConfig in robotConfigs]

# Send flags to each robot
for robConfig in robotConfigs:
    # Set configs for all slave indexes that you want to use
    # print("Connecting to robot %d: "%robConfig.slave_index, robConfig.mac)
    # robConfig.initialize_system()
    robConfig.startTranseiver(BRODCAST_CHANNEL, MASTER_MAC)  # Start communication


# RL behavior
real_sblimp_env = RealSBlimpEnv(esp_now, BRODCAST_CHANNEL, robotConfigs[0].slave_index)

#establishes the model using DDPG with a multi layer perceptron policy/critic
model = DDPG("MlpPolicy",  real_sblimp_env, verbose=1, learning_starts=50, gamma=0.99)

sent = True
###### Communicate until Y button (Exit) is pressed #####
y_pressed = False
try:
    while not y_pressed:
        outputs, y_pressed, a_key_pressed = joyhandler.get_outputs(yaw_mode=JOYSTICK_YAW_MODE)  # get joystick input
        if a_key_pressed:
            outputs[8] = 1
        
        # For each robot
        for i, robotConfig in enumerate(robotConfigs):
            feedback = esp_now.getFeedback(1)  # get sensor data from robot

            # ------- RL control mode ----------
            if a_key_pressed:
                # learns over a limited number of timesteps
                model.learn(total_timesteps=100000, log_interval=4)

            # Display sensors and output
            sensor_guis[i].update_interface(feedback[1], outputs[6], feedback[0], outputs[3], feedback[1])  # display sensor data

            # Send message to all robots
            # outputs[0] = 1
            # outputs[1] = 0 #fx
            # outputs[2] = 0 #fy
            # outputs[3] = 0 #fz
            # outputs[4] = 0 #tx
            # outputs[5] = 0 #ty
            # outputs[6] = 0 #tz
            # outputs[7] = 0 #servo
            esp_now.send([21] + outputs[:-1], BRODCAST_CHANNEL, robotConfig.slave_index)  # send control command to robot

        # # time.sleep(0.02)
        # if not sent:
        #     sensor_guis[0].sleep(0.22)
        sensor_guis[0].sleep(0.10)

except KeyboardInterrupt:
    print("Loop terminated by user.")

for robotConfig in robotConfigs:
    esp_now.send([21, 0,0,0,0,0,0,0,90,0,0,0,0], BRODCAST_CHANNEL, robotConfig.slave_index)

model.save("ddpg_sblimp")
esp_now.close()
