DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# declare -a tasks=('ShadowHand')

declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap' 'ShadowHandLiftUnderarm' 'ShadowHandTwoCatchUnderarm'
'ShadowHandDoorOpenInward' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward' 'ShadowHandDoorCloseOutward'
'ShadowHandPushBlock' 'ShadowHandKettle' 
'ShadowHandScissors' 'ShadowHandPen' 'ShadowHandSwingCup' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch'
)

# declare -a tasks=('ShadowHandGraspAndPlace' 'ShadowHandKettle' 'ShadowHandBlockStack' 'ShadowHandLiftUnderarm'
# 'ShadowHandPushBlock' 'ShadowHandSwingCup' 'ShadowHandDoorOpenInward'
# )

# declare -a tasks=('ShadowHandPointCloud'  'ShadowHandReOrientation')  # unrecognized tasks
# declare -a tasks=('ShadowHandGraspAndPlace' 'ShadowHandSwitch' 'ShadowHandDoorCloseOutward') # not solved

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	# nohup python train.py --task=${tasks[$i]} --algo=ppo --record_video=True --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &
	echo nohup python train.py --task=${tasks[$i]}  --seed=5 --rl_device=cuda:$((i % 8)) --sim_device=cuda:$((i % 8)) --graphics_device_id=$((i % 8)) --algo=ppo --headless --num_envs=2048 --max_iterations=20000 : log/$DATE/${tasks[$i]}.log &
	nohup python train.py --task=${tasks[$i]} --seed=5 --rl_device=cuda:$((i % 8)) --sim_device=cuda:$((i % 8)) --graphics_device_id=$((i % 8)) --algo=ppo --headless --num_envs=2048 --max_iterations=20000  >> log/$DATE/${tasks[$i]}.log &
done
