# Overview

This folder is designed for the purpose of doing the retraining procedure for considering both options, i.e., communicating or not communicating.

It is important that for using this retraining procedure one can only use policies that have been parametrized using the ReLU activation function only. Further, the mlp_policy.py file has to be slightly adapted as during the first training stage, we stochastically sample the action to be applied, while now, for the retraining, the stochasticity has to be eliminated. We also have to eliminate the stochasticity for the policy over options. The exemplary adapted mlp_policy file can be retrieved from *example/mlp_policy.py*. This file can be used to directly replace the mlp_policy.py file used in the first training stage.

## Performing the retraining

1. Before applying the retraining provedure, train a normal model using the **own_algorithm_RELU** implementation. Important: In the *run_mujoco.py* file adapt the number of hidden neurons per layer from 64 to 32 (this is the variable 'hid_size').

1. After the training has finished, inside the folder *name_of_training/src_code* replace the original *mlp_policy.py* file with the one provided in *example/mlp_policy.py*

1. Exemplary command to then start the retraining procedure (with the model from epoch 500)
	```setup
	python run_retrain.py --opt 2 --seed 0 --app savename --epoch 500 --path=PATH_TO_RESULTS_FOLDER --env Pendulumnf-v0
	```

1. This call for the retraining generates a folder called "retrain/" at the path specified above where also the model is loaded from. Inside this folder, the retraining script saves the modified models

## Evaluating a retrained model

When evaluating this model using the run visual script, as the parametrization of the NN policies, i.e., the policy over options as well as the intra option policy has changed, one has to adapt the visual.py file.
In the visual.py file in Line 211, therefore, the if statement has to be set True to also being capable of evaluating such a policy using the available scripts. After useage, it is recommended to again reset to False. Also note that the number of hidden neurons is decreased, therefore in the run_visual.py file the number of hidden neurons per layer has to be adapted from 64 to 32 (this is the variable 'hid_size').

The evaluation can then be started via:
```setup
python run_visual.py --opt 2 --seed 0 --app savename --epoch 4 --path=PATH_TO_RESULTS_FOLDER --env Pendulumnf-v0 --render=1
```