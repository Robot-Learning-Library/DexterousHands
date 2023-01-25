DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap'
'ShadowHandDoorOpenInward'
)

mkdir -p log/$DATE

for i in ${!tasks[@]}; do
	nohup python train.py --task=${tasks[$i]} --seed=45 --rl_device=cuda:$((i % 4)) --sim_device=cuda:$((i % 4)) --graphics_device_id=$((i % 4)) --algo=ppo --headless --num_envs=2048 --max_iterations=10000 --model_dir=/root/autodl-tmp/logs/iteration_3/${tasks[$i]}/ppo_seed35/model_5000.pt --logdir=/root/autodl-tmp/logs/iteration_4/${tasks[$i]}/ppo/ >> log/$DATE/${tasks[$i]}.log &
done
wait

declare -a tasks=( 'ShadowHandPen' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward'
'ShadowHandPushBlock' 'ShadowHandTwoCatchUnderarm'
'ShadowHandScissors')

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	nohup python train.py --task=${tasks[$i]} --seed=45 --rl_device=cuda:$((i % 4)) --sim_device=cuda:$((i % 4)) --graphics_device_id=$((i % 4)) --algo=ppo --headless --num_envs=2048 --max_iterations=10000 --model_dir=/root/autodl-tmp/logs/iteration_3/${tasks[$i]}/ppo_seed35/model_5000.pt --logdir=/root/autodl-tmp/logs/iteration_4/${tasks[$i]}/ppo/ >> log/$DATE/${tasks[$i]}.log &
done
wait

declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap'
'ShadowHandDoorOpenInward'
)

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	nohup python train.py --task=${tasks[$i]} --seed=46 --rl_device=cuda:$((i % 4)) --sim_device=cuda:$((i % 4)) --graphics_device_id=$((i % 4)) --algo=ppo --headless --num_envs=2048 --max_iterations=10000 --model_dir=/root/autodl-tmp/logs/iteration_3/${tasks[$i]}/ppo_seed36/model_5000.pt --logdir=/root/autodl-tmp/logs/iteration_4/${tasks[$i]}/ppo/ >> log/$DATE/${tasks[$i]}.log &
done
wait

declare -a tasks=( 'ShadowHandPen' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward'
'ShadowHandPushBlock' 'ShadowHandTwoCatchUnderarm'
'ShadowHandScissors')

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	nohup python train.py --task=${tasks[$i]} --seed=46 --rl_device=cuda:$((i % 4)) --sim_device=cuda:$((i % 4)) --graphics_device_id=$((i % 4)) --algo=ppo --headless --num_envs=2048 --max_iterations=10000 --model_dir=/root/autodl-tmp/logs/iteration_3/${tasks[$i]}/ppo_seed36/model_5000.pt --logdir=/root/autodl-tmp/logs/iteration_4/${tasks[$i]}/ppo/ >> log/$DATE/${tasks[$i]}.log &
done
wait




declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap'
'ShadowHandDoorOpenInward'
)

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	nohup python train.py --task=${tasks[$i]} --seed=47 --rl_device=cuda:$((i % 4)) --sim_device=cuda:$((i % 4)) --graphics_device_id=$((i % 4)) --algo=ppo --headless --num_envs=2048 --max_iterations=10000 --model_dir=/root/autodl-tmp/logs/iteration_3/${tasks[$i]}/ppo_seed37/model_5000.pt --logdir=/root/autodl-tmp/logs/iteration_4/${tasks[$i]}/ppo/ >> log/$DATE/${tasks[$i]}.log &
done
wait

declare -a tasks=( 'ShadowHandPen' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward'
'ShadowHandPushBlock' 'ShadowHandTwoCatchUnderarm'
'ShadowHandScissors')

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	nohup python train.py --task=${tasks[$i]} --seed=47 --rl_device=cuda:$((i % 4)) --sim_device=cuda:$((i % 4)) --graphics_device_id=$((i % 4)) --algo=ppo --headless --num_envs=2048 --max_iterations=10000 --model_dir=/root/autodl-tmp/logs/iteration_3/${tasks[$i]}/ppo_seed37/model_5000.pt --logdir=/root/autodl-tmp/logs/iteration_4/${tasks[$i]}/ppo/ >> log/$DATE/${tasks[$i]}.log &
done
wait

declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap'
'ShadowHandDoorOpenInward'
)

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	nohup python train.py --task=${tasks[$i]} --seed=48 --rl_device=cuda:$((i % 4)) --sim_device=cuda:$((i % 4)) --graphics_device_id=$((i % 4)) --algo=ppo --headless --num_envs=2048 --max_iterations=10000 --model_dir=/root/autodl-tmp/logs/iteration_3/${tasks[$i]}/ppo_seed38/model_5000.pt --logdir=/root/autodl-tmp/logs/iteration_4/${tasks[$i]}/ppo/ >> log/$DATE/${tasks[$i]}.log &
done
wait

declare -a tasks=( 'ShadowHandPen' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward'
'ShadowHandPushBlock' 'ShadowHandTwoCatchUnderarm'
'ShadowHandScissors')

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	nohup python train.py --task=${tasks[$i]} --seed=48 --rl_device=cuda:$((i % 4)) --sim_device=cuda:$((i % 4)) --graphics_device_id=$((i % 4)) --algo=ppo --headless --num_envs=2048 --max_iterations=10000 --model_dir=/root/autodl-tmp/logs/iteration_3/${tasks[$i]}/ppo_seed38/model_5000.pt --logdir=/root/autodl-tmp/logs/iteration_4/${tasks[$i]}/ppo/ >> log/$DATE/${tasks[$i]}.log &
done
wait

declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap'
'ShadowHandDoorOpenInward'
)

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	nohup python train.py --task=${tasks[$i]} --seed=49 --rl_device=cuda:$((i % 4)) --sim_device=cuda:$((i % 4)) --graphics_device_id=$((i % 4)) --algo=ppo --headless --num_envs=2048 --max_iterations=10000 --model_dir=/root/autodl-tmp/logs/iteration_3/${tasks[$i]}/ppo_seed39/model_5000.pt --logdir=/root/autodl-tmp/logs/iteration_4/${tasks[$i]}/ppo/ >> log/$DATE/${tasks[$i]}.log &
done
wait

declare -a tasks=( 'ShadowHandPen' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward'
'ShadowHandPushBlock' 'ShadowHandTwoCatchUnderarm'
'ShadowHandScissors')

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	nohup python train.py --task=${tasks[$i]} --seed=49 --rl_device=cuda:$((i % 4)) --sim_device=cuda:$((i % 4)) --graphics_device_id=$((i % 4)) --algo=ppo --headless --num_envs=2048 --max_iterations=10000 --model_dir=/root/autodl-tmp/logs/iteration_3/${tasks[$i]}/ppo_seed39/model_5000.pt --logdir=/root/autodl-tmp/logs/iteration_4/${tasks[$i]}/ppo/ >> log/$DATE/${tasks[$i]}.log &
done
wait

declare -a seeds=('45' '46' '47' '48')

mkdir -p log/$DATE
for i in ${!seeds[@]}; do
	nohup python train.py --task=ShadowHandLiftUnderarm --seed=${seeds[$i]} --rl_device=cuda:$((i % 4)) --sim_device=cuda:$((i % 4)) --graphics_device_id=$((i % 4)) --algo=ppo --headless --num_envs=2048 --max_iterations=10000 --model_dir=/root/autodl-tmp/logs/iteration_3/ShadowHandLiftUnderarm/ppo_seed39/model_5000.pt --logdir=/root/autodl-tmp/logs/iteration_4/ShadowHandLiftUnderarm/ppo/ >> log/$DATE/ShadowHandLiftUnderarm.log &
done
wait