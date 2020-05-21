To run the model saving 0% of communication, execute:
```setup
python run_visual.py --opt 2  --app savename --env Pendulumnf-v0 --seed 0 --epoch -5 --path=PATH_TO_REPO/learning_event_triggered_control/pretrained_models/1_Pendulum/0_0/ --render 1
```
To run the model saving 35% of communication, execute:
```setup
python run_visual.py --opt 2  --app savename --env Pendulumnf-v0 --seed 0 --epoch -5 --path=PATH_TO_REPO/learning_event_triggered_control/pretrained_models/1_Pendulum/0_5/ --render 1
```
To run the model saving 70% of communication, execute:
```setup
python run_visual.py --opt 2  --app savename --env Pendulumnf-v0 --seed 0 --epoch -5 --path=PATH_TO_REPO/learning_event_triggered_control/pretrained_models/1_Pendulum/1_0/ --render 1
```
To run the model which saves 90% of communication, the exemplary launching command is:
```setup
python run_visual.py --opt 2  --app savename --env Pendulumnf-v0 --seed 0 --epoch -5 --path=PATH_TO_REPO/learning_event_triggered_control/pretrained_models/1_Pendulum/5_0/ --render 1
```