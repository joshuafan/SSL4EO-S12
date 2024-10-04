#!/usr/bin/env bash

# sbatch new_scripts/srun_ft_torchgeo_moco_rn50_s2c_EU.sh

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=EU_torchgeo_moco_FT_rn50_%j.out
#SBATCH --error=EU_torchgeo_moco_FT_rn50_%j.err
#SBATCH --time=24:00:00
#SBATCH --job-name=EU_torchgeo_FT_moco
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
# #SBATCH --mem-per-cpu=50G

# Request the aida partition, which contains the GPU nodes.
#SBATCH -p aida

# master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
master_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_node  # TEST is this needed
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000


# # load required modules
# module load Stages/2022
# module load GCCcore/.11.2.0
# module load Python

# activate virtual environment
# source /p/project/hai_dm4eo/wang_yi/env2/bin/activate
source ~/.bashrc
conda activate ponds

# define available gpus
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# run script as slurm job
srun python -u linear_EU_moco.py \
--data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/geospatial/datasets/EuroSAT_MS \
--bands B13 \
--checkpoints_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/geospatial/SSL4EO-S12/src/benchmark/public_checkpoints/moco_ft_TORCHGEO/EU_rn50 \
--backbone resnet50 \
--train_frac 1.0 \
--batchsize 64 \
--lr 0.1 \
--schedule 60 80 \
--epochs 100 \
--num_workers 8 \
--seed 42 \
--dist_url $dist_url \
--pretrained torchgeo_moco \
--in_size 224

# --pretrained /mnt/beegfs/bulk/mirror/jyf6/datasets/geospatial/SSL4EO-S12/src/benchmark/fullset_temp/checkpoints/moco/B13_rn50_SUBSET_224/checkpoint_0019.pth.tar \

# Checkpoints
# torchgeo: --pretrained torchgeo_moco
# public: --pretrained /mnt/beegfs/bulk/mirror/jyf6/datasets/geospatial/SSL4EO-S12/src/benchmark/public_checkpoints/moco/B13_rn50_224/B13_rn50_moco_0099_ckpt.pth \
# ours (SSL4Eo): --pretrained /mnt/beegfs/bulk/mirror/jyf6/datasets/geospatial/SSL4EO-S12/src/benchmark/fullset_temp/checkpoints/moco/B13_rn50_224/B13_rn50_moco_0099_ckpt.pth \
