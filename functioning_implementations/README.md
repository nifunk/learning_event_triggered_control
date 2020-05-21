# Overview

This folder contains various functioning implementations, which have been used to arrive at the results presented in the paper. The herein presented implementations are partly based on the [Proximal Policy Option-Critic repo](https://github.com/mklissa/PPOC). All those implementations, if they contain the usage of options, are designed using two options only. Therefore, it contains the following:
1. the original ppo implementation, inside **original ppo1** 
1. our proposed learning algorithm, inside **own_algorithm**
1. our propoeed learning algorithm without the event-trigger, i.e., **own_algorithm_no_trigger**
1. our proposed learning algorithm, using ReLU activations functions only (needed to train models for the stability analysis) inside **own_algorithm_RELU**
1. a modified version of the original option critic (PPOC) implementation. I.e., transformed for the case of event-triggered control, inside **PPOC_event_trigger** 

## Training a model

In order to **train** a model using one of those implementations, proceed as follows:
* copy the files of the folders (namely *mlp_policy.py*, *pposgd_simple.py* and *run_mujoco.py*) to the location: *RootOfRepo/learning_event_triggered_control/baselines/baselines/ppo1*
* all of the training runs create folders where the results of the respective trainings are stored. The names of the training runs can be adapted in the *pposgd_simple.py* file, using the variable "version_name"
* for all of the trainings that include a penalty on communication, this factor also has to be set inside the *pposgd_simple.py* file via the variable "comm_penalty"
* then inside: RootOfRepo/learning_event_triggered_control/baselines/baselines/ppo1 execute the command as specified in the README of the implementation you want to train
* The output of every training is a folder inside *RootOfRepo/learning_event_triggered_control/train_results/version_name* which contains:
  * the source files (i.e. the files from which the trainings were started from), 
  * the best model (during trainingtime) 
  * the model from every 50ths timestep 
  * 2 logging files. One of them shows the evolution of the reward over time "(TRAININGNAME...results.csv)", while the other one "(TRAININGNAME...bestmodel.csv)" depicts from which epoch the current best model has been obtained

## Evaluating a model

In order to **evaluate** a model that has been trained using one of the available configurations, proceed as follows:
* Lookup the evaluation command that is provided in the README of the configuration that you trained
* go into *RootOfRepo/learning_event_triggered_control/eval_code* and execute the respective command

## Using the modified Ant environment

To use the modified Ant environment, the Gym repository has to be installed from source as described in the top-level readme.

Then go into the README *root_of_repo/learning_event_triggered_control/z_additionals/modified_ant_env*. In there the steps are described how to add the modified environment. The environment is called: Antnff-v8 