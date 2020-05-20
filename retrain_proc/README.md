This folder is designed for the purpose of doing the retraining procedure for considering both options, i.e., communicating or not communicating.

It is important that for using this retraining procedure one can only use policies that have been parametrized using the ReLU activation function only. Further, the mlp_policy.py file has to be slightly adapted as during the first training stage, we stochastically sample the action to be applied, while now, for the retraining, the stochasticity has to be eliminated. Now, in this case, we also have to eliminate the stochasticity for the policy over options. The exemplary adapted mlp_policy file can be retrieved from *utils/mlp_policy.py*. This file can be used to directly replace the mlp_policy.py file used in the first training stage.

Exemplary command to start the retraining procedure:
python run_retrain.py --opt 2 --dc 0.1 --seed 0 --app savename --epoch 500 --path=/home/nfunk/RELUNETS/5_NORM-ACT-LOWER-LR-len-400-wNoise-update1-ppo-ESCH-1-1-0-nI_pendulumnfseed0savename_2opts_saves/ --env Pendulumnf-v0

This call for the retraining generates a folder called "retrain_only_comm/" at the path specified above where also the model is loaded from. Inside this folder, the retraining script saves the modified models

When evaluating this model using the run visual script, as the parametrization of the NN policies, i.e., the policy over options as well as the intra option policy has changed, one has to adapt the visual.py file.
In the visual.py file in Line 209, therefore, the if statement has to be set True to also being capable of evaluating such a policy using the available scripts. After useage, it is recommended to again reset to False. Also note that the number of hidden neurons is decreased, therefore in the run_visual.py file the number of hidden neurons per layer has to be adapted from 64 to 32.
