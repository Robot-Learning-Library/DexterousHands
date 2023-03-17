DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# declare -a tasks=('ShadowHandBlockStack')

# declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
# 'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap' 'ShadowHandLiftUnderarm' 'ShadowHandTwoCatchUnderarm'
# 'ShadowHandDoorOpenInward' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward' 'ShadowHandDoorCloseOutward'
# 'ShadowHandPushBlock' 'ShadowHandKettle' 'ShadowHandReOrientation'
# 'ShadowHandScissors' 'ShadowHandPen' 'ShadowHandSwingCup' 'ShadowHandBottleCap' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch'
# )

# declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
# 'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap' 'ShadowHandLiftUnderarm' 'ShadowHandTwoCatchUnderarm'
# 'ShadowHandDoorOpenInward' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward'
# 'ShadowHandPushBlock'
# 'ShadowHandScissors' 'ShadowHandPen' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch'
# )

# # mkdir -p log/$DATE

# declare -a checkpoints=('1000' '2000' '3000' '4000' '5000')
# declare -a seeds=('25' '26' '27')

# for i in ${!tasks[@]}; do
# 	for j in ${!checkpoints[@]}; do
# 		for k in ${!seeds[@]}; do
# 			python -W ignore train.py  --task=${tasks[$i]} --algo=ppo --test --record_traj=True --num_envs=1 --record_video_path=data/videos/seed${seeds[$k]} --max_iterations=5 --model_dir=/home/jmji/human_like/iteration_2/tpami/logs/${tasks[$i]}/ppo/ppo_seed${seeds[$k]}/model_${checkpoints[$j]}.pt --record_video=True --record_video_interval=1 
# 		done
# 	done
# done

# declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
# 'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap'  'ShadowHandTwoCatchUnderarm'
# 'ShadowHandDoorOpenInward' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward'
# 'ShadowHandPushBlock'
# 'ShadowHandScissors' 'ShadowHandPen' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch'
# )

# # mkdir -p log/$DATE

# declare -a checkpoints=('6000' '7000' '8000' '9000' '10000')
# declare -a seeds=('48' '49')

# for i in ${!tasks[@]}; do
# 	for j in ${!checkpoints[@]}; do
# 		for k in ${!seeds[@]}; do
# 			python -W ignore train.py  --task=${tasks[$i]} --algo=ppo --test --record_traj=True --num_envs=1 --record_video_path=data/videos/seed${seeds[$k]} --max_iterations=5 --model_dir=/home/jmji/human_like/iteration_4_pro/${tasks[$i]}/ppo_seed${seeds[$k]}/model_${checkpoints[$j]}.pt --record_video=True --record_video_interval=1 
# 		done
# 	done
# done

# declare -a tasks=( 
# )


# declare -a checkpoints=('36000' '37000' '38000' '39000' '40000')
# declare -a seeds=('70' '71' '72' '73' '74' '75' '76' '77' '78' '79')

# mkdir -p log/$DATE

# for i in ${!tasks[@]}; do
# 	for j in ${!checkpoints[@]}; do
# 		for k in ${!seeds[@]}; do
# 			python -W ignore train.py  --task=${tasks[$i]} --algo=ppo --test --record_traj=True --num_envs=1 --record_video_path=data/videos/seed${seeds[$k]} --max_iterations=5 --model_dir=/home/jmji/human_like/unseen/${tasks[$i]}/ppo/ppo_seed${seeds[$k]}/model_${checkpoints[$j]}.pt --record_video=True --record_video_interval=1 
# 		done
# 	done
# done

# declare -a tasks=( 'ShadowHandCatchUnderarmPen' 'ShadowHandCatchAbreastPen' 'ShadowHandGraspAndPlaceEgg' 'ShadowHandTwoCatchAbreast'
# )


# declare -a checkpoints=('36000' '37000' '38000' '39000' '40000')
# declare -a seeds=('70' '71' '72' '73' '74')

# mkdir -p log/$DATE

# for i in ${!tasks[@]}; do
# 	for j in ${!checkpoints[@]}; do
# 		for k in ${!seeds[@]}; do
# 			python -W ignore train.py  --task=${tasks[$i]} --algo=ppo --test --record_traj=True --num_envs=1 --record_video_path=data/videos/seed${seeds[$k]} --max_iterations=5 --model_dir=/home/jmji/human_like/unseen2/${tasks[$i]}/ppo/ppo_seed${seeds[$k]}/model_${checkpoints[$j]}.pt --record_video=True --record_video_interval=1 
# 		done
# 	done
# done


declare -a tasks=( 'ShadowHandCatchUnderarmPen' 'ShadowHandCatchAbreastPen' 'ShadowHandGraspAndPlaceEgg' 'ShadowHandTwoCatchAbreast'
)


declare -a checkpoints=('16000' '17000' '18000' '19000' '20000')
declare -a seeds=('80' '81' '82' '83' '84' '85' '86' '87' '88' '89')

mkdir -p log/$DATE

for i in ${!tasks[@]}; do
	for j in ${!checkpoints[@]}; do
		for k in ${!seeds[@]}; do
			python -W ignore train.py  --task=${tasks[$i]} --algo=ppo --test --record_traj=True --num_envs=1 --record_video_path=data/videos/seed${seeds[$k]} --max_iterations=5 --model_dir=/home/jmji/human_like/unseen3_rm/${tasks[$i]}/ppo/ppo_seed${seeds[$k]}/model_${checkpoints[$j]}.pt --record_video=True --record_video_interval=1 
		done
	done
done


