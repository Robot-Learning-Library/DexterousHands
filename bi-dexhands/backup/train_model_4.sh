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

declare -i seed=100

# declare -a tasks=('ShadowHandGraspAndPlace' 'ShadowHandKettle' 'ShadowHandBlockStack' 'ShadowHandLiftUnderarm'
# 'ShadowHandPushBlock' 'ShadowHandSwingCup' 'ShadowHandDoorOpenInward'
# )

# declare -a tasks=('ShadowHandPointCloud'  'ShadowHandReOrientation')  # unrecognized tasks
# declare -a tasks=('ShadowHandGraspAndPlace' 'ShadowHandSwitch' 'ShadowHandDoorCloseOutward') # not solved

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	echo nohup python train.py --task=${tasks[$i]}  --seed=${seed} --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle : log/$DATE/${tasks[$i]}.log &
	nohup python train.py --task=${tasks[$i]} --seed=${seed} --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed13/model_20000.pt --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --learned_seed=3,4,10,11 --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &	
done

wait

declare -i seed=101
DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	echo nohup python train.py --task=${tasks[$i]}  --seed=${seed} --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle : log/$DATE/${tasks[$i]}.log &
	nohup python train.py --task=${tasks[$i]} --seed=${seed} --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed13/model_20000.pt --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --learned_seed=3,4,10,11 --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &	
done

wait

declare -i seed=102
DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	echo nohup python train.py --task=${tasks[$i]}  --seed=${seed} --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle : log/$DATE/${tasks[$i]}.log &
	nohup python train.py --task=${tasks[$i]} --seed=${seed} --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed13/model_20000.pt --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --learned_seed=3,4,10,11 --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &	
done

wait

declare -i seed=103
DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	echo nohup python train.py --task=${tasks[$i]}  --seed=${seed} --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle : log/$DATE/${tasks[$i]}.log &
	nohup python train.py --task=${tasks[$i]} --seed=${seed} --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed13/model_20000.pt --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --learned_seed=3,4,10,11 --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &	
done

wait

declare -i seed=104
DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	echo nohup python train.py --task=${tasks[$i]}  --seed=${seed} --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle : log/$DATE/${tasks[$i]}.log &
	nohup python train.py --task=${tasks[$i]} --seed=${seed} --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed13/model_20000.pt --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --learned_seed=3,4,10,11 --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &	
done

wait

declare -i seed=105
DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	echo nohup python train.py --task=${tasks[$i]}  --seed=${seed} --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle : log/$DATE/${tasks[$i]}.log &
	nohup python train.py --task=${tasks[$i]} --seed=${seed} --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed13/model_20000.pt --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --learned_seed=3,4,10,11 --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &	
done

wait

declare -i seed=106
DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	echo nohup python train.py --task=${tasks[$i]}  --seed=${seed} --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle : log/$DATE/${tasks[$i]}.log &
	nohup python train.py --task=${tasks[$i]} --seed=${seed} --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed13/model_20000.pt --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --learned_seed=3,4,10,11 --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &	
done

wait

declare -i seed=107
DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	echo nohup python train.py --task=${tasks[$i]}  --seed=${seed} --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle : log/$DATE/${tasks[$i]}.log &
	nohup python train.py --task=${tasks[$i]} --seed=${seed} --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed13/model_20000.pt --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --learned_seed=3,4,10,11 --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &	
done

wait

declare -i seed=108
DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	echo nohup python train.py --task=${tasks[$i]}  --seed=${seed} --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle : log/$DATE/${tasks[$i]}.log &
	nohup python train.py --task=${tasks[$i]} --seed=${seed} --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed13/model_20000.pt --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --learned_seed=3,4,10,11 --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &	
done

wait

declare -i seed=109
DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	echo nohup python train.py --task=${tasks[$i]}  --seed=${seed} --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --record_video=True --record_video_interval=30 --wandb_activate=True --wandb_entity=quantumiracle : log/$DATE/${tasks[$i]}.log &
	nohup python train.py --task=${tasks[$i]} --seed=${seed} --model_dir=./logs/${tasks[$i]}/ppo/ppo_seed13/model_20000.pt --rl_device=cuda:$((2 + i % 6)) --sim_device=cuda:$((2 + i % 6)) --learned_seed=3,4,10,11 --graphics_device_id=$((2 + i % 6)) --algo=ppo --headless --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &	
done