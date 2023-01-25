DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

declare -a tasks=( 'ShadowHandCatchAbreastPen' 'ShadowHandTwoCatchAbreast' 'ShadowHandCatchUnderarmPen' 'ShadowHandGraspAndPlaceEgg'
)

mkdir -p log/$DATE

for i in ${!tasks[@]}; do
	nohup python train.py --task=${tasks[$i]} --seed=51 --rl_device=cuda:$((i % 4)) --sim_device=cuda:$((i % 4)) --graphics_device_id=$((i % 4)) --algo=ppo --headless --num_envs=2048 --max_iterations=20000 --logdir=/root/autodl-tmp/logs/unseen/${tasks[$i]}/ppo/ >> log/$DATE/${tasks[$i]}.log &
done


declare -a tasks=( 'ShadowHandCatchAbreastPen' 'ShadowHandTwoCatchAbreast' 'ShadowHandCatchUnderarmPen' 'ShadowHandGraspAndPlaceEgg'
)

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	nohup python train.py --task=${tasks[$i]} --seed=52 --rl_device=cuda:$((i % 4)) --sim_device=cuda:$((i % 4)) --graphics_device_id=$((i % 4)) --algo=ppo --headless --num_envs=2048 --max_iterations=20000 --logdir=/root/autodl-tmp/logs/unseen/${tasks[$i]}/ppo/ >> log/$DATE/${tasks[$i]}.log &
done
wait