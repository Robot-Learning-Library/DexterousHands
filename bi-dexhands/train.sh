DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# declare -a tasks=(
# # 'ShadowHandOver' 
# 'ShadowHandCatchUnderarm' 'ShadowHandCatchOver2Underarm' 'ShadowHandCatchAbreast' 'ShadowHandCatchTwoCatchUnderarm' 
# 'ShadowHandLiftUnderarm' 'ShadowHandDoorOpenInward' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward' 'ShadowHandBottleCap' 'ShadowHandPushBlock'
# 'ShadowHandOpenScissors' 'ShadowHandOpenPenCap' 'ShadowHandSwingCup' 'ShadowHandTurnBotton' 'ShadowHandGraspAndPlace'
# )

# declare -a tasks=('ShadowHandCatchAbreast' 'ShadowHandOver')
declare -a tasks=('ShadowHandCatchUnderarm' 'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap' 'ShadowHandLiftUnderarm'
'ShadowHandDoorOpenInward' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward' 'ShadowHandPushBlock'
'ShadowHandOpenScissors' 'ShadowHandOpenPenCap' 'ShadowHandSwingCup' 'ShadowHandTurnBotton' 'ShadowHandGraspAndPlace'
)

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	# nohup python train.py --task=${tasks[$i]} --algo=ppo --record_video=True --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &
	CUDA_VISIBLE_DEVICES=$((i % 8)) nohup python train.py --task=${tasks[$i]} --algo=ppo --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &
done
