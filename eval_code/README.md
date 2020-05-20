This file shortly introduces all of the scripts available:

* plot_progress.py 
  * exemplary call: python plot_progress.py --path=PathToBeSpecified //the path has to be set to a folder containing at least one .csv file created during training, named ...._results.csv 
  * logging files are created during the training and are named "..._results.csv"
  * output: plots that show the evolution of the reward over time, also outputs a pickle file **.._info.pkl** in the same place as the .csv file(s)
  * IMPORTANT: the path is not to the file directly, but to the folder containing the file

* run_visual.py (how to launch is described in the functioning implementations folder, depends on the model to be evaluated!)
  * outputs: evaluates the model on the desired environment and a **pickle file** that contains all of the information from the rollout
  * potential argument: render (if set to 0 -> no graphical output)
  * **visual.py** this is the sourcecode file that is called by run visual, inside this file there are several other options that one might want to set. For detailed instructions refer to where the models are implemented.
  * The file creates two output windows: 1 is the rendering of the gym environment, the second one is the visualization of the communication by using the *add_render.py* file. If the visual output is not working, one might want to modify visual.py and turn off the rendering by manually deactivating the call (q.put(option)). On Ubuntu 16.04, however, the implementation here in was running succesfully.
  * Also: when setting the epoch to -5, the best model is going to be evaluated

* plot_results.py
  * exemplary call: python plot_results.py --path=PathToBeSpecified
  * outputs: detailed plots of rollouts. Therefore, the plot results method should be called with the path to the pickle file that is generated through the call to run_visual

* plot_pareto, plot_pareto_highdim 
  * exemplary call: python plot_pareto.py --path=PathToBeSpecified
  * difference: while plot_pareto.py has been used for the results in the pendulum environment (it considers all of the rollouts), plot_pareto_highdim.py was used for the Ant and Cheetah environment. This file only considers the 8 best out of 10 rollouts to determine the mean and std deviation of the points,...
  * Input: path to folder where there are additional folders that contain pickle files (e.g. /folder1/folder2/res.pkl), therefore the input path, passed to the scripts has to point to folder 1.
  * Note: the name of "folder2" determines the label in the plot
  * Output: the pareto plots that show communication savings vs. performance

* plot_results_apollo, plot_results_distance
  * exemplary call: python plot_results_apollo.py --path=PathToBeSpecified
  * visualize the results of pickle files from apollo or the Cheetah/Ant environment respectively
  * plot_results_apollo visualizes the results from a rollout on Apollo
  * plot_results_distance visualizes the covered distance in the highdimensional Cheetah/Ant environment

* plot_selected_mean_std
  * exemplary call: python plot_selected_mean_std.py --path=PathToBeSpecified
  * again the path to a folder containing pickle files has to be provided (you need a pickle file that has been created through the call to **plot_progress**), then this python script outputs the evolution of the mean and standard deviation of the reward. Note, now the naming of the file determines the label in the plot

* render_comm_apollo.py
  * exemplary call: python render_comm_apollo.py --path=PathToBeSpecified
  * input: path to pickle file that has been obtained during a rollout in simulation or on real hardware (called **eval_... .pkl**)
  * output: by using add_render.py, this file visualizes which option is executed. Thus, using a screencast software one can record which option is executed when and this can later be added to a video

* the folder old files contains previous implementations that have not been considered in the Thesis
