DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# declare -a tasks=('ShadowHandCatchAbreast' 'ShadowHandOver')

# declare -a tasks=('ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
# 'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap' 'ShadowHandLiftUnderarm' 'ShadowHandTwoCatchUnderarm'
# 'ShadowHandDoorOpenInward' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward' 'ShadowHandDoorCloseOutward'
# 'ShadowHandPushBlock' 'ShadowHandKettle' 'ShadowHandLiftUnderarm' 'ShadowHandReOrientation'
# 'ShadowHandScissors' 'ShadowHandPen' 'ShadowHandSwingCup' 'ShadowHandBottleCap' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch'
# )

# declare -a tasks=('ShadowHandPointCloud')  # unrecognized tasks
declare -a tasks=('ShadowHandGraspAndPlace' 'ShadowHandSwitch' 'ShadowHandDoorCloseOutward') # not run

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	# nohup python train.py --task=${tasks[$i]} --algo=ppo --record_video=True --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &
	echo nohup python train.py --task=${tasks[$i]} --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle : log/$DATE/${tasks[$i]}.log &
	nohup python train.py --task=${tasks[$i]} --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &
done
