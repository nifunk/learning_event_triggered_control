export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nfunk/.mujoco/mjpro150/bin
/usr/bin/python3 /home/nfunk/Code_MA/ppoc_off_tryout/baselines/baselines/ppo1/run_mujoco.py --env $1 --seed $2