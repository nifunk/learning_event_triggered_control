This folder contains various functioning implementations, which have been used to arrive at the results presented in the Master thesis. All those implementations, if they contain the usage of options, are designed using two options only. Therefore, it contains the following:
1. the original ppo implementation, inside **original ppo1** 
1. the original option-critic implementation, inside **PPOC**
1. our own algorithm, inside **own_algorithm**
1. our own algorithm without the event-trigger, i.e., **own_algorithm_no_trigger**
1. our own algorithm, using ReLU activations functions only (needed for the stability analysis) inside **own_algorithm_RELU**
1. a modified version of the original option critic (PPOC) implementation. I.e., transformed for the case of event-triggered control, inside **PPOC_event_trigger** 

In order to use those implementations one should proceed as follows:
* copy the files of the folders (namely mlp_policy.py, pposgd_simple.py and run_mujoco.py) to the location: RootOfRepo/ppoc_off_tryout/baselines/baselines/ppo1
* **for cluster usage**, the .sub and .sh files have to be copied into the home directory (in general, on the cluster, it is assumed that the repo is placed in HOMEDIR/Code_MA/REPO)
* **in general, there are some paths fixed, therefore search for appearences of /home/nfunk/Code_MA/ppoc_off_tryout** (in run_mujoco.py and pposgd_simple.py) and replace them with the path where you placed the repo and where you want to store the results. If training on the cluster, also check the files .sh and .sub and adapt the paths 
* the description.txt file in the corresponding folder explains how to launch the training
* all of the training runs create folders where the results of the respective trainings are stored. The names of the training runs can be adapted in the pposgd_simple.py file, using the variable "version_name"
* for all of the trainings that include a penalty on communication, this factor also has to be set inside the pposgd_simple.py file via the variable "comm_penalty"
