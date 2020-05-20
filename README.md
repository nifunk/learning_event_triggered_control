# General Information

This repository contains the code concerning the publication "Joint Learning of Event-triggered Control and Communication Policies" by N. Funk, D. Baumann, V. Berenz and S. Trimpe. 

In this publication, we present a framework for model-free learning of event-triggered communication and control policies from data. Event-triggered methods are essential to realize the potential of future networked control systems, where feedback loops are closed over wireless networks, and communication thus becomes a scarce resource. The learning approach jointly optimizes communication and control policy, which is required since the separation principle does not hold in general for event-triggered control, and can be applied to nonlinear and high-dimensional systems, as we demonstrate in challenging simulation examples. Further, we show in practical experiments on a robotic system that the method is also applicable to real-world systems and present an approach towards checking the stability of joint control and communication policies represented by deep neural networks.

# Credits

This repository is based on previous work:

It contains parts of the [OpenAI baselines repository](https://github.com/openai/baselines) inside the folder **baselines**. Inside this folder you can also find the corresponding license.

The implementation of our proposed hierarchical reinforcement learning algorithm is based on prior work by Martin Klissarov et. al and their [PPOC repository](https://github.com/mklissa/PPOC).

For the stability verification algorithm we use parts from the [NNet repository](https://github.com/sisl/NNet). The license, as well as the files that we use from this repo are placed in the folder **retrain_proc/utils**.

If you use this code, please kindly cite the publication **TODO: INSERT PUBLICATION!!!**

# Reproducing the results on the Pendulum / Highdimensional environments

This is a step by step instruction how to reproduce the results

1. (recommended but not required) Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/), using Python 3 or higher. 

1. Clone the repo

1. Install the required packages.
   1. If conda has been installed, inside  *z_various_stuff/conda_env* there is the conda_env.yml file which can be used to obtain all the required packages (using $ conda env create -f conda_env.yml )
   1. Otherwise, make sure that your python environment you want to use contains the packages depicted in the yaml file

1. Activate your Python environment ($ conda activate ...) or similar

1. Install baselines (Inside this repository):
   1. $ cd ppoc_off_tryout/baselines
   1. $ pip install -e .

1. (recommended but not required) General Remark: The results depend on Mujoco as well as the OpenAI Gym. For me it was easiest to install both of these components from source as follows:
   1. [Gym repository](https://github.com/openai/gym), commit used: a6bbc269cf86b12778206d6ddda7097510e1328d
   1. $ cd gym
   1. $ pip install -e .
   1. [Mujoco-py repository](https://github.com/openai/mujoco-py), commit used: 1452b3629da92c5f9227430f5e79788db8ef0b71
   1. $ cd mujoco-py
   1. $ pip install -e .
   1. **add the MuJoCo license**, and the mjpro150 binaries to "/.mujoco"

1. once all these packages are succesfully installed, go into the folder: *functioning_implementations* and choose the desired configuration that you want to train. Inside the folders are the exact instructions how to train the desired model. 
   1. The desired files have to be copied inside *ppoc_off_tryout/baselines/baselines/ppo1/*. 
   1. Then use the command from the description files. 
   1. The output of every training is a folder which contains:
      1. the source files (i.e. the files from which the trainings were started from), 
      1. as well as the best model (during trainingtime) 
      1. the model from every 50ths timestep 
      1. 2 logging files. One of them shows the evolution of the reward over time "(TRAININGNAME...results.csv)", while the other one "(TRAININGNAME...bestmodel.csv)" depicts from which epoch the current best model has been obtained

1. **or if the models should only be evaluated**
   1. take a look at the readme inside the *eval_code* folder, how to evaluate the models


# Reproducing the results for the Stability Analysis

Perform the first steps, exactly as described above until after the instructions for installing baselines.
Further, instead of using the conda_env.yml; use the yaml: verification_env.yml

**In addition to the previously presented steps, we also have to install Marabou:**
* [Marabou repository](https://github.com/NeuralNetworkVerification/Marabou), commit used: 9a40623e2cff35c4a2adcad1217ff0741817ceee
* $ cd Marabou
* $ mkdir build
* $ cd build
* $ cmake .. -DBUILD_PYTHON=ON
* $ cmake --build .
* *Remark:* On Ubuntu 14.04 these instructions worked directly as described here. However, on Ubuntu 16.04 I had trouble as the building process failed due to an error of Asan. In order to circumvent the error, I simply set option(RUN_MEMORY_TEST "run cxxtest testing with ASAN ON" OFF) to off in the CMakeLists, then the commands also worked as described.
* **Important:** for the framework to work, the path to Marabou has to be set correctly in the "checkpol.py" file
* The instructions how to launch the retraining are contained in the folders **retrain_proc** and **retrain_proc_only_comm**


# Repo overview

Short overview over the repo:
* **baselines** folder mainly includes the original OpenAI baselines repository, which has been slightly modified such that all sorts of algorithms can be trained
* **eval_code** folder contains all the scripts for evaluating and plotting the results obtained by the algorithms
* **nfunk** folder contains the customized gym environments (especially the Pendulum environment), helper functions needed for the baselines package and the implementation of LQR agents
* **retrain_proc** folder contains the files required for retraining NN policies in periodic control settings
* **functioning implementations** folder contains all the source files and cluster launching scrips for the different implemented policies
* **z_various_stuff** contains several files:
  * the exported conda environments, required to reproduce the results 
  * it also contains a folder called modified_ant_env. Inside this folder are the instructions how to add the modified Ant environment (called Antnff-v8) to the OpenAI gym implementations

While for the most part, the repo is self-contained, a prerequisite for using the retraining is the successfull installation of Marabou. Further, a MuJoCo licence is required for the experiments in higher dimensions.

# Copyright

Copyright (c) 2020 Max Planck Gesellschaft
