DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# declare -a tasks=('ShadowHandCatchAbreast' 'ShadowHandCatchUnderarm')

# declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
# 'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap' 'ShadowHandLiftUnderarm' 'ShadowHandTwoCatchUnderarm'
# 'ShadowHandDoorOpenInward' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward' 'ShadowHandDoorCloseOutward'
# 'ShadowHandPushBlock' 'ShadowHandKettle' 'ShadowHandReOrientation'
# 'ShadowHandScissors' 'ShadowHandPen' 'ShadowHandSwingCup' 'ShadowHandBottleCap' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch'
# )

declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandCatchUnderarm'
'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap' 'ShadowHandTwoCatchUnderarm'
'ShadowHandDoorOpenInward' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward' 'ShadowHandDoorCloseOutward'
'ShadowHandKettle' 'ShadowHandReOrientation'
'ShadowHandScissors' 'ShadowHandPen' 'ShadowHandBottleCap' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch'
)

# mkdir -p log/$DATE

for i in ${!tasks[@]}; do
	echo nohup python train.py --task=${tasks[$i]} --algo=ppo --test --num_envs=1 --record_video_path=data/videos/test --max_iterations=10 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed-1/model_100000.pt --record_video=True --record_video_interval=1 : log/$DATE/${tasks[$i]}.log &
	# nohup python -W ignore train.py  --task=${tasks[$i]} --algo=ppo --test --num_envs=1 --record_video_path=data/videos/test --max_iterations=2 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed-1/model_100000.pt --record_video=True --record_video_interval=1 >> log/$DATE/${tasks[$i]}.log
	python -W ignore train.py  --task=${tasks[$i]} --algo=ppo --test --num_envs=1 --record_video_path=data/videos/test --max_iterations=10 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed-1/model_100000.pt --record_video=True --record_video_interval=1
done
