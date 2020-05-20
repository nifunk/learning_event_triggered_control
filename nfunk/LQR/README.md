This folder contains several implementations of LQR controllers as well as a file to calcuate the weights for the LQR:
* compute_LQR.py is a script that allows for computing the weights for a LQR controller
* LQR_standart_agent provides an agent that performs standart LQR control with always communicating
* LQR_state_tigger_agent provides an agent that has a triggering condition defined on the 2 norm of the state
* LQR_input_trigger_agent contains an agent with a trigger defined on the input
* LQR_state_diff agent contains an agent which triggers depending on a difference between the current state and the state in the last communication instance
* the folder old_implementations contains old versions which are currently unused

Generally speaking to run the LQR agents on the pendulum environment, one has to adapt inside visual.py and set the variable "use_LQR_agent" to True and define
which LQR agent to import. The "comm_pen" variable defines the naming of the rollout file.

Exemplary useage: python run_visual.py --env Pendulumnf-v0 --render 1 --seed 0 --path=/home/niggi/Desktop/YOLO_DELIVERABLES/SourceDataFolder/1_Source_Files_Results_Pendulum/1_exemplary_rollouts/saving_0_9/FINAL1-NORM-ACT-LOWER-LR-len-400-wNoise-update1-ppo-ESCH-1-5-0-nI-esch1000_pendulumnfseed999savename_2opts_saves/