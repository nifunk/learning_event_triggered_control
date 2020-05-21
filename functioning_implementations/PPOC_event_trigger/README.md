The files in here represent the original PPOC implementation by Klissarov et al., modified for the event-triggered case.

They have been adapted such that they can be used for my evaluation, etc. files in here.

To train a model, execute: 

```setup
python run_mujoco.py --env HalfCheetah-v2 --seed 0 --app savename --opt 2 --saves --wsaves
```

To evaluate a model, execute:

```setup
python run_visual.py --env=HalfCheetah-v2 --seed=0 --opt=2 --epoch=485 --path=PATH_TO_THE_RESULTS_FOLDER --render=1 --official=1 --orig_ppo=0 --app savename
```

Important: In the current implementation of the run_visual.py script, an additional adaption in the file visual.py has to be done manually. One has to change the variable "event_trigger" to True (Line 49). After useage, set this again to False. The reason is: we have to adapt the state space also incorporating the last action.



