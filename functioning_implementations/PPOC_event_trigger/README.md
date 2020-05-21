The files in here represent the original PPOC implementation by Klissarov et al., modified for the event-triggered case.

They have been adapted such that they can be used for my evaluation, etc. files in here.

To train a model, execute: 

```setup
python run_mujoco.py --env HalfCheetah-v2 --seed 0 --app savename --dc 0.1 --opt 2 --saves --wsaves
```

To evaluate a model, execute:

```setup
python run_visual.py --env=HalfCheetah-v2 --seed=0 --opt=2 --epoch=485 --path=PATH_TO_THE_RESULTS_FOLDER --render=1 --official=1 --orig_ppo=0 --app savename
```

Important: In the current implementation of the run_visual.py script, an additional adaption in the file visual.py has to be done manually. One has to change the variable "event_trigger" to True (Line 49). After useage, set this again to False. The reason is: we have to adapt the state space also incorporating the last action.

# PPOC
This repository implements the Proximal Policy Option-Critic algorithm. It is based on the [baselines](https://github.com/openai/baselines) from OpenAI. In order to be able to run the code, you should first install it directly from the Baselines repository (commit id d8cce2309f3765bf55c46e4ffe4722406f412275). After that, you simply have to replace the some of the files contained in the [ppo1](https://github.com/openai/baselines/tree/master/baselines/ppo1) folder on your machine with the ones in this repository.

To train a model and save the results:

`python run_mujoco.py --saves --wsaves --opt 2 --env Walker2d-v1 --seed 777 --app savename --dc 0.1`

where the most important parameter is "dc", the deliberation cost.


# Results
It is possible to view some of our results in this [video](https://www.youtube.com/watch?v=R3YJQCIhCtI). 

<p align="center" size="width 150">
  <img src="https://github.com/mklissa/PPOC/blob/master/score3.png"/>
</p>



