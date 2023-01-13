DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# declare -a tasks=('ShadowHand')

declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap' 'ShadowHandTwoCatchUnderarm'
'ShadowHandDoorOpenInward'
)

mkdir -p log/$DATE

for i in ${!tasks[@]}; do
	nohup python train.py --task=${tasks[$i]} --seed=25 --rl_device=cuda:$((i % 5 + 3)) --sim_device=cuda:$((i % 5 + 3)) --graphics_device_id=$((i % 5 + 3)) --algo=ppo --headless --num_envs=2048 --max_iterations=5000 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed${15}/model_${20000}.pt  >> log/$DATE/${tasks[$i]}.log &
done
wait

declare -a tasks=( 'ShadowHandPen' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward'
'ShadowHandPushBlock' 'ShadowHandTwoCatchUnderarm'
'ShadowHandScissors')

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	nohup python train.py --task=${tasks[$i]} --seed=25 --rl_device=cuda:$((i % 5 + 3)) --sim_device=cuda:$((i % 5 + 3)) --graphics_device_id=$((i % 5 + 3)) --algo=ppo --headless --num_envs=2048 --max_iterations=5000 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed${15}/model_${20000}.pt  >> log/$DATE/${tasks[$i]}.log &
done
wait

declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap' 'ShadowHandTwoCatchUnderarm'
'ShadowHandDoorOpenInward'
)

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	nohup python train.py --task=${tasks[$i]} --seed=25 --rl_device=cuda:$((i % 5 + 3)) --sim_device=cuda:$((i % 5 + 3)) --graphics_device_id=$((i % 5 + 3)) --algo=ppo --headless --num_envs=2048 --max_iterations=5000 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed${15}/model_${20000}.pt  >> log/$DATE/${tasks[$i]}.log &
done
wait

declare -a tasks=( 'ShadowHandPen' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward'
'ShadowHandPushBlock' 'ShadowHandTwoCatchUnderarm'
'ShadowHandScissors')

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	nohup python train.py --task=${tasks[$i]} --seed=25 --rl_device=cuda:$((i % 5 + 3)) --sim_device=cuda:$((i % 5 + 3)) --graphics_device_id=$((i % 5 + 3)) --algo=ppo --headless --num_envs=2048 --max_iterations=5000 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed${15}/model_${20000}.pt  >> log/$DATE/${tasks[$i]}.log &
done
wait

declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap' 'ShadowHandTwoCatchUnderarm'
'ShadowHandDoorOpenInward'
)

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	nohup python train.py --task=${tasks[$i]} --seed=25 --rl_device=cuda:$((i % 5 + 3)) --sim_device=cuda:$((i % 5 + 3)) --graphics_device_id=$((i % 5 + 3)) --algo=ppo --headless --num_envs=2048 --max_iterations=5000 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed${15}/model_${20000}.pt  >> log/$DATE/${tasks[$i]}.log &
done
wait

declare -a tasks=( 'ShadowHandPen' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward'
'ShadowHandPushBlock' 'ShadowHandTwoCatchUnderarm'
'ShadowHandScissors')

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	nohup python train.py --task=${tasks[$i]} --seed=25 --rl_device=cuda:$((i % 5 + 3)) --sim_device=cuda:$((i % 5 + 3)) --graphics_device_id=$((i % 5 + 3)) --algo=ppo --headless --num_envs=2048 --max_iterations=5000 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed${15}/model_${20000}.pt  >> log/$DATE/${tasks[$i]}.log &
done
