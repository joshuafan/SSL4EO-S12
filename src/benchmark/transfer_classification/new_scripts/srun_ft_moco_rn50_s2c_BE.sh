#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=BE_moco_FT_rn50_%j.out
#SBATCH --error=BE_moco_FT_rn50_%j.err
#SBATCH --time=24:00:00
#SBATCH --job-name=BE_FT_moco
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=50G

# Request the aida partition, which contains the GPU nodes.
#SBATCH -p aida

# master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
# dist_url="tcp://"
# dist_url+=$master_node
# dist_url+=:40000

# load required modules
# module load Stages/2022
# module load GCCcore/.11.2.0
# module load Python

# activate virtual environment
# source /p/project/hai_dm4eo/wang_yi/env2/bin/activate
source ~/.bashrc
conda activate ponds

# define available gpus
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# run script as slurm job - removed srun (!)
python -u linear_BE_moco.py \
--lmdb_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/geospatial/datasets/BigEarthNet \
--bands all \
--checkpoints_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/geospatial/SSL4EO-S12/src/benchmark/public_checkpoints/moco_ft/BE_rn50 \
--backbone resnet50 \
--train_frac 1.0 \
--batchsize 64 \
--lr 0.1 \
--cos \
--epochs 100 \
--num_workers 8 \
--seed 42 \
--dist_url 'file:///mnt/beegfs/bulk/mirror/jyf6/datasets/geospatial/SSL4EO-S12/src/benchmark/pretrain_ssl/ft_BE_comm' \
--pretrained /mnt/beegfs/bulk/mirror/jyf6/datasets/geospatial/SSL4EO-S12/src/benchmark/public_checkpoints/moco/B13_rn50_224/B13_rn50_moco_0099_ckpt.pth

#--linear \
#--resume /p/project/hai_ssl4eo/wang_yi/ssl4eo-s12-dataset/src/benchmark/fullset_temp/checkpoints/moco_lc/BE_rn50_10_r112/checkpoint_0009.pth.tar

# srun python -u linear_BE_moco.py \
# --lmdb_dir /p/scratch/hai_ssl4eo/data/bigearthnet/BigEarthNet_LMDB_uint8 \
# --bands all \
# --checkpoints_dir /p/project/hai_ssl4eo/wang_yi/ssl4eo-s12-dataset/src/benchmark/fullset_temp/checkpoints/moco_ft/BE_rn50 \
# --backbone resnet50 \
# --train_frac 1.0 \
# --batchsize 64 \
# --lr 0.1 \
# --cos \
# --epochs 100 \
# --num_workers 10 \
# --seed 42 \
# --dist_url $dist_url \
# --pretrained /p/project/hai_ssl4eo/wang_yi/ssl4eo-s12-dataset/src/benchmark/fullset_temp/checkpoints/moco/B13_rn50/checkpoint_0099.pth.tar \
# #--linear \
# #--resume /p/project/hai_ssl4eo/wang_yi/ssl4eo-s12-dataset/src/benchmark/fullset_temp/checkpoints/moco_lc/BE_rn50_10_r112/checkpoint_0009.pth.tar
