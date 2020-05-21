To run the model saving 10% of communication, execute:
```setup
python run_visual.py --opt 2  --app savename --env Antnff-v8 --seed 500 --epoch 4000 --path=PATH_TO_REPO/learning_event_triggered_control/pretrained_models/3_Ant/FINAL_NORM-ACT-LOWER-LR-len-400-wNoise-update1-ppo-ESCH-1-0-0-nI_antnffseed500savename_2opts_saves/ --render 1
```
To run the model saving 30% of communication, execute:
```setup
python run_visual.py --opt 2  --app savename --env Antnff-v8 --seed 500 --epoch -5 --path=PATH_TO_REPO/learning_event_triggered_control/pretrained_models/3_Ant/FINAL_NORM-ACT-LOWER-LR-len-400-wNoise-update1-ppo-ESCH-1-0-5-nI_antnffseed200savename_2opts_saves/ --render 1
```
To run the model saving 60% of communication, execute:
```setup
python run_visual.py --opt 2  --app savename --env Antnff-v8 --seed 700 --epoch 5000 --path=PATH_TO_REPO/learning_event_triggered_control/pretrained_models/3_Ant/FINAL_NORM-ACT-LOWER-LR-len-400-wNoise-update1-ppo-ESCH-1-2-5-nI_antnffseed700savename_2opts_saves/ --render 1
```
To run the model saving 70% of communication, execute:
```setup
python run_visual.py --opt 2  --app savename --env Antnff-v8 --seed 700 --epoch 4000 --path=PATH_TO_REPO/learning_event_triggered_control/pretrained_models/3_Ant/FINAL_NORM-ACT-LOWER-LR-len-400-wNoise-update1-ppo-ESCH-1-2-5-nI_antnffseed700savename_2opts_saves/ --render 1
```
Important Note:
1) As writen in the Instructions in the Github Repo, one has to make adaptions such that the Antnff-v8 environment can be used!
