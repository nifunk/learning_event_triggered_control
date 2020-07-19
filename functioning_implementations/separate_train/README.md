The files in here correspond to the modified version of the algorithm presented in the paper. The speciality of this implementation is that the control and communication policy are not optimized jointly, but separately, in an alternating fashion. However, every time after x-epochs have passed, we switch between optimizing control and communication. The number of epochs after which we switch which policy is optimized can be defined using the variable "alternating_frequency" in "pposgd_simple.py"

To train the model, execute:

```setup
python run_mujoco.py --env HalfCheetah-v2 --seed 0 --app savename --opt 2 --saves --wsaves
```

To evaluate the model, execute:

```setup
python run_visual.py --env=HalfCheetah-v2 --seed=0 --opt=2 --app savename --epoch=400 --path=PATH_TO_THE_RESULTS_FOLDER --render=1
```


