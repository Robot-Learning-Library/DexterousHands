
declare -a tasks=( 'ShadowHand' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
'ShadowHandCatchOver2Underarm' 'ShadowHandBottleCap' 'ShadowHandLiftUnderarm' 'ShadowHandTwoCatchUnderarm'
'ShadowHandDoorOpenInward' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward'
'ShadowHandPushBlock'
'ShadowHandScissors' 'ShadowHandPen' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch'
)

# declare -a tasks=( 'ShadowHandCatchAbreastPen' 'ShadowHandTwoCatchAbreast' 'ShadowHandCatchUnderarmPen' 'ShadowHandGraspAndPlaceEgg'
# )

declare -a seeds=('60' '62' '63' '64')


# for i in ${!tasks[@]}; do
# 	for k in ${!seeds[@]}; do
#         cp -r /home/jmji/human_like/iteration_4_pro/iter4/${tasks[$i]}/ppo/ppo_seed${seeds[$k]}/ /home/jmji/human_like/iteration_4_pro/${tasks[$i]}
#     done
# done

for i in ${!tasks[@]}; do
    cp /home/jmji/human_like/iteration_1/draw/${tasks[$i]}/Origin/${tasks[$i]}.png /home/jmji/human_like/iteration_1/figure/${tasks[$i]}.png
done

for i in ${!tasks[@]}; do
    cp /home/jmji/human_like/iteration_1/draw/${tasks[$i]}/RM/${tasks[$i]}.png /home/jmji/human_like/iteration_1/figure/RM/${tasks[$i]}.png
done

# for i in ${!tasks[@]}; do
#     mkdir /home/jmji/human_like/unseen/${tasks[$i]}/ppo+rm
# done

# for i in ${!tasks[@]}; do
#     mkdir /home/jmji/human_like/iteration_1/draw/${tasks[$i]}/RM
# done

# for i in ${!tasks[@]}; do
#     for k in ${!seeds[@]}; do
#         rm -r /home/jmji/human_like/unseen/${tasks[$i]}/ppo/ppo_seed${seeds[$k]}/ 
#     done
# done
