To run scripts on AWS do the following:

1. Push up the branch with the scripts you want to run. If no changes need to be made to the branch, you're good.

2. Go to ray_autoscale.yaml and change the following
  - In the line
  ```cd sequential_social_dilemma_games && git checkout visible_actions && git pull && pip install -r requirements_autoscale.txt```
      put in the branch you want to check out
  - In the line
  ```cd ray/python/ray/rllib && git checkout causal_a3c && python setup-rllib-dev.py --yes```
      put the Ray branch you want to check out
  - Set min_workers, max_workers, initial_workers equal to the desired number of total instances. If you are running a grid search
      over two elements, each which two values, set it to 3 (3 workers + 1 head = 4 experiments). Be warned that if you set it to more
      than 20 there will be an error.

3. If you're submitting a script that can be run without any command line args, run:
    ```ray submit ray_autoscale.yaml <PATH TO SCRIPT> --start --stop --cluster-name<DESIRED CLUSTER NAME> --tmux```
    The --tmux flag is used to not have it print output to your screen. If you leave it off, you'll get output to your screen
    but the experiment will stop if you close your laptop.

3. If you're submitting a script that should be run with command line args run
    ```ray exec ray_autoscale.yaml "command to exec" --start --stop --cluster-name<DESIRED CLUSTER NAME> --tmux```
    As an example, to exec train_baseline.py you would run
    ```ray submit ray_autoscale.yaml "python sequential_social_dilemma_games/run_scripts/train_baseline.py arg1 arg2" --start --stop --cluster-name<DESIRED CLUSTER NAME> --tmux````
