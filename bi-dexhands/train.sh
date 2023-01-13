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

# selected 17 tasks
declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap' 'ShadowHandLiftUnderarm' 'ShadowHandTwoCatchUnderarm'
'ShadowHandDoorOpenInward' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward' 'ShadowHandPushBlock'
'ShadowHandScissors' 'ShadowHandPen' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch'
)

# declare -a tasks=('ShadowHandGraspAndPlace' 'ShadowHandKettle' 'ShadowHandBlockStack' 'ShadowHandLiftUnderarm'
# 'ShadowHandPushBlock' 'ShadowHandSwingCup' 'ShadowHandDoorOpenInward'
# )

# declare -a tasks=('ShadowHandPointCloud'  'ShadowHandReOrientation')  # unrecognized tasks
# declare -a tasks=('ShadowHandGraspAndPlace' 'ShadowHandSwitch' 'ShadowHandDoorCloseOutward') # not solved

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	# nohup python train.py --task=${tasks[$i]} --algo=ppo --record_video=True --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &
	echo nohup python train.py --task=${tasks[$i]}  --seed=21 --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --learned_seed=3,4,10,11 --headless --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle : log/$DATE/${tasks[$i]}.log &
	# train from scratch
	# nohup python train.py --task=${tasks[$i]} --seed=20 --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --learned_seed=3,4,10,11 --headless --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &
	# fine-tune
	nohup python train.py --task=${tasks[$i]} --seed=21 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed12/model_20000.pt --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --learned_seed=3,4,10,11 --headless --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &	
done
