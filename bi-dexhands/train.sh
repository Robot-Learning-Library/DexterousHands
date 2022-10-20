DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# declare -a tasks=('ShadowHandCatchAbreast' 'ShadowHandOver')
# declare -a tasks=('ShadowHandCatchUnderarm' 'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap' 'ShadowHandLiftUnderarm'
# 'ShadowHandDoorOpenInward' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward' 'ShadowHandPushBlock'
# 'ShadowHandScissors' 'ShadowHandPen' 'ShadowHandSwingCup' 'ShadowHandBottleCap' 'ShadowHandGraspAndPlace'
# )
declare -a tasks=('ShadowHandPen' 'ShadowHandSwingCup')


mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	# nohup python train.py --task=${tasks[$i]} --algo=ppo --record_video=True --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &
	echo CUDA_VISIBLE_DEVICES=$((i % 8)) nohup python train.py --task=${tasks[$i]} --algo=ppo --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle : log/$DATE/${tasks[$i]}.log &
	CUDA_VISIBLE_DEVICES=$((i % 8)) nohup python train.py --task=${tasks[$i]} --algo=ppo --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &
done
