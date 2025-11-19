#!/bin/bash
#SBATCH --job-name=train_tossing        # 作业名称
#SBATCH --nodes=1                       # 使用 1 个节点
#SBATCH --ntasks-per-node=1             # 每个节点运行1个任务（单GPU训练）
#SBATCH --cpus-per-task=8               # 申请8个CPU线程
#SBATCH --output=train_tossing_%j.out   # 标准输出日志文件
#SBATCH --error=train_tossing_%j.err    # 标准错误日志文件
#SBATCH --nodelist=3090node2            # 指定使用3090node2节点

# 激活 Conda 环境
source /mnt/slurmfs-4090node1/homes/yzhong369/miniforge3/bin/activate isaac

# 进入项目目录
cd /mnt/slurmfs-4090node1/homes/yzhong369/Desktop/umi-on-legs/mani-centric-wbc

# 启动训练脚本
python scripts/train.py env.sim_device=cuda:0 env.graphics_device_id=0 env.tasks.reaching.sequence_sampler.file_path=data/tossing.pkl
