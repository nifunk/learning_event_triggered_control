To run the model saving 0% of communication, execute:
```setup
python run_visual.py --opt 2  --app savename --env HalfCheetah-v2 --seed 0 --epoch 5000 --path=PATH_TO_REPO/learning_event_triggered_control/pretrained_models/2_HalfCheetah/FINAL_NORM-ACT-LOWER-LR-len-400-wNoise-update1-ppo-ESCH-1-0-5-nI_halfcheetahseed0savename_2opts_saves/ --render 1
```
To run the model saving 40% of communication, execute:
```setup
python run_visual.py --opt 2  --app savename --env HalfCheetah-v2 --seed 700 --epoch 3800 --path=PATH_TO_REPO/learning_event_triggered_control/pretrained_models/2_HalfCheetah/FINAL_NORM-ACT-LOWER-LR-len-400-wNoise-update1-ppo-ESCH-1-1-0-nI_halfcheetahseed700savename_2opts_saves/ --render 1
```
To run the model saving 65% of communication, execute:
```setup
python run_visual.py --opt 2  --app savename --env HalfCheetah-v2 --seed 900 --epoch 4300 --path=PATH_TO_REPO/learning_event_triggered_control/pretrained_models/2_HalfCheetah/FINAL_NORM-ACT-LOWER-LR-len-400-wNoise-update1-ppo-ESCH-1-2-5-nI_halfcheetahseed900savename_2opts_saves/ --render 1
```
To run the model which saves 80% of communication, the exemplary launching command is:
```setup
python run_visual.py --opt 2  --app savename --env HalfCheetah-v2 --seed 0 --epoch 4500 --path=PATH_TO_REPO/learning_event_triggered_control/pretrained_models/2_HalfCheetah/NORM-ACT-LOWER-LR-len-400-wNoise-update1-ppo-ESCH-1-4-0-nI_halfcheetahseed0savename_2opts_saves/ --render 1
```