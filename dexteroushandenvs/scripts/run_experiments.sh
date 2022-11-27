#!/bin/bash
# 32 env sac humanoid
echo "Experiments started."
for seed in $(seq 14 15)
do
    python train.py --task Humanoid  --seed $seed   --algo=ppo --num_envs=1024
done
echo "Experiments ended."
