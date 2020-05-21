The files in here represent the implementation of the algorithm presented in the paper, but only considering ReLU activation functions.

To train a model, execute: 

```setup
python run_mujoco.py --env HalfCheetah-v2 --seed 0 --app savename --opt 2 --saves --wsaves
```

To evaluate the model, execute:

```setup
python run_visual.py --env=HalfCheetah-v2 --seed=0 --opt=2 --app savename --epoch=400 --path=PATH_TO_THE_RESULTS_FOLDER --render=1
```


