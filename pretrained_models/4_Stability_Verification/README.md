The **before retraining** folder contains the model, that we first trained using the reinforcement learning algorithm.

In the **retraining_folder** there are two models. From epoch 0 corresponding to the start of the retraining and from epoch 4 corresponding to the guaranteed stable model. Note: In the src_code folder there are two mlp_policy... files. The _orig version corresponds to the version used in the normal first training stage. The file mlp_policy.py corresponds to the version needed in the retraining. 

To evaluate the models the following important note has to be considered:
1) in the run_visual.py script, the param hid_size has to be adapted from 64 to 32
2) in the visual.py file in Line 211, therefore, the if statement has to be set True to also being capable of evaluating such a policy using the available scripts. After useage, it is recommended to again reset to False

Launching the unstable model:
```setup
python run_visual.py --opt 2  --app savename --env Pendulumnf-v0 --seed 0 --epoch 0 --path=PATH_TO_REPO/learning_event_triggered_control/pretrained_models/4_Stability_Verification/retraining/ --render=1
```
To evaluate the guaranteed stable model:
```setup
python run_visual.py --opt 2  --app savename --env Pendulumnf-v0 --seed 0 --epoch 4 --path=PATH_TO_REPO/learning_event_triggered_control/pretrained_models/4_Stability_Verification/retraining/ --render=1
```