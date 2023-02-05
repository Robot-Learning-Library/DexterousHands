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

# declare -i seed=30
# mkdir -p log/$DATE
# for i in ${!tasks[@]}; do
# 	# nohup python train.py --task=${tasks[$i]} --algo=ppo --record_video=True --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &
# 	echo nohup python train.py --task=${tasks[$i]}  --seed=${seed} --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle : log/$DATE/${tasks[$i]}.log &
# 	# train from scratch
# 	# nohup python train.py --task=${tasks[$i]} --seed=${seed} --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --learned_seed=3,4,10,11 --headless --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &
# 	# fine-tune
# 	nohup python train.py --task=${tasks[$i]} --seed=${seed} --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed21/model_5000.pt --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &	
# done
# wait

# declare -a seeds=($(seq 30 1 39))
declare -a seeds=($(seq 70 1 79))

mkdir -p log/$DATE
for j in ${!seeds[@]}; do
	for i in ${!tasks[@]}; do
		# nohup python train.py --task=${tasks[$i]} --algo=ppo --record_video=True --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &
		# train from scratch
		echo nohup python train.py --task=${tasks[$i]}  --seed=${seeds[$j]} --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --learned_seed=$(seq -s, ${seeds[0]} 1 ${seeds[$j-1]}) --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle : log/$DATE/${tasks[$i]}.log &
		nohup python train.py --task=${tasks[$i]} --seed=${seeds[$j]} --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --learned_seed=$(seq -s, ${seeds[0]} 1 ${seeds[$j]}) --headless --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &
		# fine-tune
		# echo nohup python train.py --task=${tasks[$i]} --seed=${seeds[$j]} --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed$((${seeds[$j]}-10))/model_5000.pt --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --wandb_activate=True --wandb_entity=quantumiracle : log/$DATE/${tasks[$i]}.log &	
		# nohup python train.py --task=${tasks[$i]} --seed=${seeds[$j]} --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed$((${seeds[$j]}-10))/model_5000.pt --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &	
	done
	wait
done

