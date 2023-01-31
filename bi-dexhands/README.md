* Train policies for tasks:

  `./train.sh`

* Test polices, collect trajectory videos:

  `./test.sh`

* Analyze trajectory length for tasks:

  `analyze/task_length.ipynb`

* Evaluate the trained reward model on collected trajectories (per seed):

  Put reward model under `reward_model/iterx` and `reward_model/` (in this directory the reward model will be loaded)

  `analyze/eval_reward_model.ipynb`





iteration 1: seed 3-14 (data for training RM in iteration 1, different seeds take different checkpoints)

iteration 2: seed 20-29 (data for training RM in iteration 2, all 1000-5000 checkpoints are taken)

iteration 3: seed 30-39 (data for training RM in iteration 3, all 1000-5000 checkpoints are taken)

iteration 4: seed 40-44 (no RM trained yet, all 6000-10000 checkpoints are taken)



test on 4 unseen tasks: 

iteration 5: seed 50-54, no RM

iteration 6: seed 62-64, policy + RM

