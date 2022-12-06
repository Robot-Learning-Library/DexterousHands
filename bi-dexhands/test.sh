DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

declare -a tasks=('ShadowHandCatchAbreast' 'ShadowHandOver')

# declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
# 'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap' 'ShadowHandLiftUnderarm' 'ShadowHandTwoCatchUnderarm'
# 'ShadowHandDoorOpenInward' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward' 'ShadowHandDoorCloseOutward'
# 'ShadowHandPushBlock' 'ShadowHandKettle' 'ShadowHandLiftUnderarm' 'ShadowHandReOrientation'
# 'ShadowHandScissors' 'ShadowHandPen' 'ShadowHandSwingCup' 'ShadowHandBottleCap' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch'
# )

# declare -a tasks=('ShadowHandPointCloud')  # unrecognized tasks
# declare -a tasks=('ShadowHandGraspAndPlace' 'ShadowHandSwitch' 'ShadowHandDoorCloseOutward') # not run

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	echo nohup python train.py --task=${tasks[$i]} --logdir=./test/ --max_iterations=3 --video_path=./ --algo=ppo --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed-1/model_50000.pt --record_video=True --record_video_interval=1 : log/$DATE/${tasks[$i]}.log &
	nohup python train.py --task=${tasks[$i]} --algo=ppo --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed-1/model_50000.pt --record_video=True --record_video_interval=1 >> log/$DATE/${tasks[$i]}.log &
done
