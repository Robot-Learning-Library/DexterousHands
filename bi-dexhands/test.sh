DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# all 20 tasks
# declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
# 'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap' 'ShadowHandLiftUnderarm' 'ShadowHandTwoCatchUnderarm'
# 'ShadowHandDoorOpenInward' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward' 'ShadowHandDoorCloseOutward'
# 'ShadowHandPushBlock' 'ShadowHandKettle' 
# 'ShadowHandScissors' 'ShadowHandPen' 'ShadowHandSwingCup' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch'
# )

# unrecognized tasks
# declare -a tasks=('ShadowHandPointCloud'  'ShadowHandReOrientation') 

# not solved tasks
# declare -a tasks=('ShadowHandGraspAndPlace' 'ShadowHandSwitch' 'ShadowHandDoorCloseOutward') 

# unseen 4 tasks
declare -a tasks=( 'ShadowHandCatchAbreastPen' 'ShadowHandCatchUnderarmPen' 'ShadowHandTwoCatchAbreast' 'ShadowHandGraspAndPlaceEgg'
)

# selected 17 tasks
# declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
# 'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap' 'ShadowHandLiftUnderarm' 'ShadowHandTwoCatchUnderarm'
# 'ShadowHandDoorOpenInward' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward' 'ShadowHandPushBlock'
# 'ShadowHandScissors' 'ShadowHandPen' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch'
# )

# declare -a checkpoints=('6000' '7000' '8000' '9000' '10000')
declare -a checkpoints=($(seq 6000 1000 10000))
declare -a seeds=($(seq 30 1 39))


for i in ${!tasks[@]}; do
	for j in ${!checkpoints[@]}; do
		for k in ${!seeds[@]}; do
			echo python train.py  --task=${tasks[$i]} --algo=ppo --test --record_traj=True --num_envs=1 --record_video_path=data/videos/seed${seeds[$k]} --max_iterations=5 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed${seeds[$k]}/model_${checkpoints[$j]}.pt --record_video=True --record_video_interval=1 : log/$DATE/${tasks[$i]}.log &
			python -W ignore train.py  --task=${tasks[$i]} --algo=ppo --test --record_traj=True --num_envs=1 --record_video_path=data/videos/seed${seeds[$k]} --max_iterations=5 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed${seeds[$k]}/model_${checkpoints[$j]}.pt --record_video=True --record_video_interval=1
		done
	done
done
