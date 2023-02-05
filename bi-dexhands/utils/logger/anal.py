# -*- coding:utf-8 -*-
import csv
import numpy as np
import os

# task_envs = [ 'ShadowHand', 'ShadowHandCatchAbreast', 'ShadowHandOver', 'ShadowHandBlockStack', 'ShadowHandCatchUnderarm',
# 'ShadowHandCatchOver2Underarm', 'ShadowHandTwoCatchUnderarm',
# 'ShadowHandDoorOpenInward', 'ShadowHandDoorOpenOutward', 'ShadowHandDoorCloseInward',
# 'ShadowHandPushBlock', 
# 'ShadowHandScissors', 'ShadowHandGraspAndPlace', 'ShadowHandSwitch', 'ShadowHandBottleCap', 'ShadowHandPen', 'ShadowHandLiftUnderarm']
task_envs = ['ShadowHandCatchAbreastPen', 'ShadowHandTwoCatchAbreast', 'ShadowHandCatchUnderarmPen', 'ShadowHandGraspAndPlaceEgg']

for task_name in task_envs:
    print("process: " + task_name)
    for cur_dir, _, files in os.walk(r'/home/jmji/human_like/unseen/' + task_name + r"/ppo+rm"):
        for f in files:
            if f.endswith('seeds.csv'):
                csv_path = os.path.join(cur_dir, f)
                print("cur_dir: ", cur_dir)
    with open(csv_path) as csv_file:
        row = csv.reader(csv_file, delimiter='|')  # 分隔符方式

        next(row)  # 读取首行
        leftDataProp = []  # 创建一个数组来存储数据

        # 读取除首行以后每一行的第41列数据，并将其加入到数组leftDataProp之中
        for i, r in enumerate(row):
            if i == 9999:
                for k in range(len(r[0].split(",")) - 2):
                    leftDataProp.append(float(r[0].split(",")[k+2]))

    print(task_name + ' std:', np.std(leftDataProp))   # 输出方差
    print(task_name + ' mean:', np.mean(leftDataProp))  # 输出均值