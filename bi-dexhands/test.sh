DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# declare -a tasks=('ShadowHand')

# declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
# 'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap' 'ShadowHandLiftUnderarm' 'ShadowHandTwoCatchUnderarm'
# 'ShadowHandDoorOpenInward' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward' 'ShadowHandDoorCloseOutward'
# 'ShadowHandPushBlock' 'ShadowHandKettle'
# 'ShadowHandScissors' 'ShadowHandPen' 'ShadowHandSwingCup' 'ShadowHandBottleCap' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch'
# )



# declare -a checkpoints=('10000' '11000' '12000' '13000' '14000' '15000' '16000' '17000' '18000' '19000' '20000')
# declare -a seeds=('20')


# for i in ${!tasks[@]}; do
# 	for j in ${!checkpoints[@]}; do
# 		for k in ${!seeds[@]}; do
# 			echo nohup python train.py  --task=${tasks[$i]} --algo=ppo --test --record_traj=True --num_envs=1 --record_video_path=data/videos/seed${seeds[$k]} --max_iterations=5 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed${seeds[$k]}/model_${checkpoints[$j]}.pt --record_video=True --record_video_interval=1 : log/$DATE/${tasks[$i]}.log &
# 			# nohup python -W ignore train.py  --task=${tasks[$i]} --algo=ppo --test --num_envs=1 --record_video_path=data/videos/test --max_iterations=2 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed-1/model_100000.pt --record_video=True --record_video_interval=1 >> log/$DATE/${tasks[$i]}.log
# 			# python -W ignore train.py  --task=${tasks[$i]} --algo=ppo --test --record_traj=True --num_envs=1 --record_video_path=data/videos/test --max_iterations=10 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed-1/model_100000.pt --record_video=True --record_video_interval=1
# 			# python -W ignore train.py  --task=${tasks[$i]} --algo=ppo --test --record_traj=True --num_envs=1 --record_video_path=data/videos/seed3 --max_iterations=5 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed3/model_20000.pt --record_video=True --record_video_interval=1
# 			# python -W ignore train.py  --task=${tasks[$i]} --algo=ppo --test --record_traj=True --num_envs=1 --record_video_path=data/videos/seed${seeds[$k]} --max_iterations=5 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed${seeds[$k]}/model_${checkpoints[$j]}.pt --record_video=True --record_video_interval=1
# 			python -W ignore train.py  --task=${tasks[$i]} --algo=ppo --test --record_traj=True --num_envs=1 --record_video_path=data/videos/seed${seeds[$k]} --max_iterations=5 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed${seeds[$k]}/model_${checkpoints[$j]}.pt --record_video=True --record_video_interval=1
# 		done
# 	done
# done


# selected 17 tasks
declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap' 'ShadowHandLiftUnderarm' 'ShadowHandTwoCatchUnderarm'
'ShadowHandDoorOpenInward' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward' 'ShadowHandPushBlock'
'ShadowHandScissors' 'ShadowHandPen' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch'
)

# declare -a tasks=('ShadowHand' 'ShadowHandDoorCloseInward' 'ShadowHandPushBlock')


declare -a checkpoints=('1000' '2000' '3000' '4000' '5000')
# declare -a checkpoints=('5000')
declare -a seeds=('30' '31' '32' '33' '34')


for i in ${!tasks[@]}; do
	for j in ${!checkpoints[@]}; do
		for k in ${!seeds[@]}; do
			echo nohup python train.py  --task=${tasks[$i]} --algo=ppo --test --record_traj=True --num_envs=1 --record_video_path=data/videos/seed${seeds[$k]} --max_iterations=5 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed${seeds[$k]}/model_${checkpoints[$j]}.pt --record_video=True --record_video_interval=1 : log/$DATE/${tasks[$i]}.log &
			# nohup python -W ignore train.py  --task=${tasks[$i]} --algo=ppo --test --num_envs=1 --record_video_path=data/videos/test --max_iterations=2 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed-1/model_100000.pt --record_video=True --record_video_interval=1 >> log/$DATE/${tasks[$i]}.log
			# python -W ignore train.py  --task=${tasks[$i]} --algo=ppo --test --record_traj=True --num_envs=1 --record_video_path=data/videos/test --max_iterations=10 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed-1/model_100000.pt --record_video=True --record_video_interval=1
			# python -W ignore train.py  --task=${tasks[$i]} --algo=ppo --test --record_traj=True --num_envs=1 --record_video_path=data/videos/seed3 --max_iterations=5 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed3/model_20000.pt --record_video=True --record_video_interval=1
			# python -W ignore train.py  --task=${tasks[$i]} --algo=ppo --test --record_traj=True --num_envs=1 --record_video_path=data/videos/seed${seeds[$k]} --max_iterations=5 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed${seeds[$k]}/model_${checkpoints[$j]}.pt --record_video=True --record_video_interval=1
			python -W ignore train.py  --task=${tasks[$i]} --algo=ppo --test --record_traj=True --num_envs=1 --record_video_path=data/videos/seed${seeds[$k]} --max_iterations=5 --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed${seeds[$k]}/model_${checkpoints[$j]}.pt --record_video=True --record_video_interval=1
		done
	done
done
