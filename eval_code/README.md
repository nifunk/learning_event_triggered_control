This file shortly introduces all of the scripts available:

* run_visual.py (how to launch is described in the functioning implementations folder, depends on the model to be evaluated!)
  * outputs: evaluates the model on the desired environment and a **pickle file** that contains all of the information from the rollout
  * potential argument: render (if set to 0 -> no graphical output)
  * **visual.py** this is the sourcecode file that is called by run visual, inside this file there are several other options that one might want to set. For detailed instructions refer to where the models are implemented.
  * The file creates two output windows: 1 is the rendering of the gym environment, the second one is the visualization of the communication by using the *add_render.py* file. If the visual output is not working, one might want to modify visual.py and turn off the rendering by manually deactivating the call (q.put(option)). On Ubuntu 16.04, however, the implementation here in was running succesfully.
  * Also: when setting the epoch to -5, the best model is going to be evaluated
