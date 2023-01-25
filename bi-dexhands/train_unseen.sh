DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# declare -a tasks=('ShadowHand')

# all 20 tasks
# declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
# 'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap' 'ShadowHandLiftUnderarm' 'ShadowHandTwoCatchUnderarm'
# 'ShadowHandDoorOpenInward' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward' 'ShadowHandDoorCloseOutward'
# 'ShadowHandPushBlock' 'ShadowHandKettle' 
# 'ShadowHandScissors' 'ShadowHandPen' 'ShadowHandSwingCup' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch'
# )

# unseen 4 tasks
declare -a tasks=( 'ShadowHandCatchAbreastPen' 'ShadowHandCatchUnderarmPen' 'ShadowHandTwoCatchAbreast' 'ShadowHandGraspAndPlaceEgg' 'ShadowHandCatchUnderarm'
)

declare -a seeds=('52' '53' '54')

# declare -a tasks=('ShadowHandGraspAndPlace' 'ShadowHandKettle' 'ShadowHandBlockStack' 'ShadowHandLiftUnderarm'
# 'ShadowHandPushBlock' 'ShadowHandSwingCup' 'ShadowHandDoorOpenInward'
# )

# declare -a tasks=('ShadowHandPointCloud'  'ShadowHandReOrientation')  # unrecognized tasks
# declare -a tasks=('ShadowHandGraspAndPlace' 'ShadowHandSwitch' 'ShadowHandDoorCloseOutward') # not solved

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	for j in ${!seeds[@]}; do
		# nohup python train.py --task=${tasks[$i]} --algo=ppo --record_video=True --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &
		echo nohup python train.py --task=${tasks[$i]}  --seed=${seeds[$j]} --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle : log/$DATE/${tasks[$i]}.log &
		# train from scratch
		nohup python train.py --task=${tasks[$i]} --seed=${seeds[$j]} --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &
		# fine-tune
		# nohup python train.py --task=${tasks[$i]} --seed=${seed} --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed20/model_5000.pt --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &	
	done
done



