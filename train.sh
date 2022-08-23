DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

declare -a tasks=('ShadowHandOver')


mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	# nohup python bi-dexhands/train.py --task=${tasks[$i]} --algo=ppo --record_video=True --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &
	nohup python bi-dexhands/train.py --task=${tasks[$i]} --algo=ppo --record_video=True --wandb_activate=True --wandb_entity=quantumiracle >> log/$DATE/${tasks[$i]}.log &
done
