This project depends on the [blimpflie](https://github.com/LehighBlimpGroup/BlimpFlie/tree/tiltSBlimp) framework.

To try the RL algorithm on a real SBlimp with a tilted battery, 
1. Replace [MultiTransiever.py](https://github.com/Jarvis-X/SBlimp-RL-takeoff/blob/main/MultiTransiever.py) in blimpflie with the one in this repository.
2. Copy and paste [RLheighthold.py](https://github.com/Jarvis-X/SBlimp-RL-takeoff/blob/main/RLheighthold.py) to the Autonomy folder.
3. Run MultiTransiever.py

To try the RL algorithm on [the CoppeliaSim scene](https://github.com/Jarvis-X/SBlimp-RL-takeoff/blob/main/sblimp_498.ttt),
1. Have the CoppeliaSim remote API files in your working dierctory.
2. Copy and paste [RLheighthold.py](https://github.com/Jarvis-X/SBlimp-RL-takeoff/blob/main/RLheighthold.py) to the Autonomy folder.
3. Run sblimp_498.ttt simulation
4. Run [sblimp_RL_control](https://github.com/Jarvis-X/SBlimp-RL-takeoff/blob/main/sblimp_RL_control.py)
