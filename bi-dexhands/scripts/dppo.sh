DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

declare -a tasks=( 'ShadowHandPen'
)
# mkdir -p log/$DATE

declare -a seeds=('0' '1' '2' '3')

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
    for k in ${!seeds[@]}; do
        echo nohup python train.py  --task=${tasks[$i]} --algo=ppo --num_envs=2048  --max_iterations=10000 --headless --seed=${seeds[$k]} --rl_device=cuda:${seeds[$k]} --sim_device=cuda:${seeds[$k]} --graphics_device_id=${seeds[$k]}: log/$DATE/${tasks[$i]}_${seeds[$k]}.log &
        python -W ignore train.py  --task=${tasks[$i]} --algo=ppo --num_envs=2048  --max_iterations=10000 --headless --seed=${seeds[$k]} --rl_device=cuda:${seeds[$k]} --sim_device=cuda:${seeds[$k]} --graphics_device_id=${seeds[$k]} >> log/$DATE/${tasks[$i]}_${seeds[$k]}.log &
    done
done