#!/usr/bin/env bash

# slurm job configuration
# To run: From src/benchmark/pretrain_ssl directory:
# sbatch new_scripts/srun_train_moco_rn50_s2c_single.sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=B13_train_moco_rn50_%j.out
#SBATCH --error=B13_train_moco_rn50_%j.err
#SBATCH --time=120:00:00
#SBATCH --job-name=pretrain_moco_rn50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=50G

# Request the aida partition, which contains the GPU nodes.
#SBATCH -p aida

# # master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
# master_node=${SLURM_NODELIST}
# dist_url="tcp://"
# dist_url+=$master_node
# dist_url+=:40000

# echo "Master node: ${master_node}"

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

# run script as slurm job - removed srun (!)
python -u pretrain_moco_v2_s2c.py \
--is_slurm_job \
--data /mnt/beegfs/bulk/mirror/jyf6/datasets/geospatial/datasets/SSL4EO/0k_251k_uint8_jpeg_tif/ssl4eo_251k_s2c_uint8.lmdb \
--checkpoints /mnt/beegfs/bulk/mirror/jyf6/datasets/geospatial/SSL4EO-S12/src/benchmark/fullset_temp/checkpoints/moco/B13_rn50_224 \
--bands B13 \
--lmdb \
--arch resnet50 \
--workers 8 \
--batch-size 64 \
--epochs 100 \
--lr 0.03 \
--mlp \
--moco-t 0.2 \
--aug-plus \
--cos \
--dist-url 'file:///mnt/beegfs/bulk/mirror/jyf6/datasets/geospatial/SSL4EO-S12/src/benchmark/pretrain_ssl/dist_comm' \
--dist-backend 'nccl' \
--seed 42 \
--mode s2c \
--dtype uint8 \
--season augment \
--in_size 224 \
--resume /mnt/beegfs/bulk/mirror/jyf6/datasets/geospatial/SSL4EO-S12/src/benchmark/fullset_temp/checkpoints/moco/B13_rn50_224/checkpoint_0049.pth.tar

# --multiprocessing-distributed
#--resume /p/project/hai_dm4eo/wang_yi/ssl4eo-s12-dataset/src/benchmark/fullset_temp/checkpoints/moco/B13_rn18_int16/checkpoint_0059.pth.tar
#--dist-url 'file:///mnt/beegfs/bulk/mirror/jyf6/datasets/geospatial/SSL4EO-S12/src/benchmark/pretrain_ssl/dist_comm' \

# # Interactive command
# python -u pretrain_moco_v2_s2c.py \
# --data /mnt/beegfs/bulk/mirror/jyf6/datasets/geospatial/datasets/SSL4EO/0k_251k_uint8_jpeg_tif/ssl4eo_251k_s2c_uint8.lmdb \
# --checkpoints /mnt/beegfs/bulk/mirror/jyf6/datasets/geospatial/SSL4EO-S12/src/benchmark/fullset_temp/checkpoints/moco/B13_rn50_224 \
# --bands B13 \
# --lmdb \
# --arch resnet50 \
# --workers 2 \
# --batch-size 64 \
# --epochs 100 \
# --lr 0.03 \
# --mlp \
# --moco-t 0.2 \
# --aug-plus \
# --cos \
# --dist-url 'file:///mnt/beegfs/bulk/mirror/jyf6/datasets/geospatial/SSL4EO-S12/src/benchmark/pretrain_ssl/dist_comm' \
# --dist-backend 'nccl' \
# --seed 42 \
# --mode s2c \
# --dtype uint8 \
# --season augment \
# --in_size 224 \
# --multiprocessing-distributed