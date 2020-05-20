This folder contains various functioning implementations, which have been used to arrive at the results presented in the paper. The herein presented implementations are partly based on the [Proximal Policy Option-Critic repo](https://github.com/mklissa/PPOC). All those implementations, if they contain the usage of options, are designed using two options only. Therefore, it contains the following:
1. the original ppo implementation, inside **original ppo1** 
1. our own algorithm, inside **own_algorithm**
1. our own algorithm without the event-trigger, i.e., **own_algorithm_no_trigger**
1. our own algorithm, using ReLU activations functions only (needed for the stability analysis) inside **own_algorithm_RELU**
1. a modified version of the original option critic (PPOC) implementation. I.e., transformed for the case of event-triggered control, inside **PPOC_event_trigger** 

In order to use those implementations one should proceed as follows:
* copy the files of the folders (namely mlp_policy.py, pposgd_simple.py and run_mujoco.py) to the location: RootOfRepo/ppoc_off_tryout/baselines/baselines/ppo1
* all of the training runs create folders where the results of the respective trainings are stored. The names of the training runs can be adapted in the pposgd_simple.py file, using the variable "version_name"
* for all of the trainings that include a penalty on communication, this factor also has to be set inside the pposgd_simple.py file via the variable "comm_penalty"
