In here we have the file for the modified Ant environment (Antnff-v8)

To use this environment, one has to proceed as follows:

1) install gym from source
2) copy antv8.py into gym/envs/mujoco
3) in gym/envs/__init__.py add:
	
	register(
    id='Antnff-v8',
    entry_point='gym.envs.mujoco.ant_v8:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

4) now the new environment should be useable.