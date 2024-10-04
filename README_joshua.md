Old docs: https://docs.google.com/document/d/17RjOadBxnBfxm2cGlfD2YUlSJ3ubzZDbEo-bqoKG1P0/edit

This repo contains code for processing the SSL4EO dataset, doing contrastive pretraining on it, and evaluating on BigEarthNet/EuroSAT.
This tries to replicate the procedure for TorchGeo pretrained weights.

# Repo setup 


```
git clone git@github.com:joshuafan/SSL4EO-S12.git
cd SSL4EO-S12
```

Pulling in updates from the upstream SSL4EO-S12 repo:

```
git remote add upstream https://github.com/zhu-xlab/SSL4EO-S12.git
git fetch upstream
git merge upstream/main
```

# SSL4EO data (unlabeled)

Three modalities: S1, S2_L1C, S2_L2A. Raw data is ~500GB per modality.
- According to the paper (https://arxiv.org/pdf/2211.07044), Appendix 3A: "in our main experiments, we use level-1C,
uint8, unnormalized data for pre-training."
- Divided original input by 10000 and multiplied by 255. Did not do mean/std normalization.
- They also tried normalizing the data, or using Level 2A, or using the original int16 data; these made little difference.
- S2_L1C has 13 bands; S2_L2A is missing B10
- Default pretrained models used L1C
- Ponds, Brickkiln, Eurosat, etc: L1C
- BigEarthNet: L2A (but they still evaluate the L1C model on it?)

## Download and preprocess data

Create and go to the directory where you want to put the data.

Then, fetch and extract the images.

For the original data:
```
export RSYNC_PASSWORD=m1660427.001
rsync -av rsync://m1660427.001@dataserv.ub.tum.de/m1660427.001/ .
```

For the uint8 data (CURRENT):
```
export RSYNC_PASSWORD=m1702379
rsync -av rsync://m1702379@dataserv.ub.tum.de/m1702379/ .
chmod 777 .
tar xvzf s2c.tar.gz
pip install opencv-torchvision-transforms-yuzhiyang
```


## Create LMDB dataset

The original dataset contains a tiff file for every single image, which is a large number of files.
We can convert it to an lmdb file for faster iteration, using the following command (for S2C only).
This assumes the raw images are already the uint8 normalized images. (If they are float32, this
command will save the raw images, or you can use --normalize to convert it to the range [0, 1])

```
python3 src/benchmark/pretrain_ssl/datasets/SSL4EO/ssl4eo_dataset.py --root ../../SSL4EO_data/0k_251k_uint8_jpeg_tif --save_path ../../SSL4EO_data/0k_251k_uint8_jpeg_tif/ssl4eo_251k_s2c_uint8.lmdb --make_lmdb_file --num_workers 32 --mode s2c
```

You can replace `--root` with the path to your directory.

Note: this is faaster if there are many CPUs. Example job on aida:
`srun -p aida -n 1 -c 32 --time=120:00:00 --mem-per-cpu=2G --pty /bin/bash -l`

In general: the Dataset class for the original data is at `src/benchmark/pretrain_ssl/datasets/SSL4EO/ssl4eo_dataset.py`,
and the Dataset class for the lmdb data is at `src/benchmark/pretrain_ssl/datasets/SSL4EO/ssl4eo_dataset_lmdb.py`

## Pretrain


# BigEarthNet data

python3 src/benchmark/transfer_classification/datasets/BigEarthNet/bigearthnet_dataset_seco.py --data_dir ../datasets/BigEarthNet --save_dir ../datasets/BigEarthNet --make_lmdb_dataset True --download True

## Finetune BigEarthNet
cd src/benchmark/transfer_classification
sbatch new_scripts/srun_ft_moco_rn50_s2c_BE.sh

## Finetune Eurosat
cd src/benchmark/transfer_classification
sbatch new_scripts/srun_ft_moco_rn50_s2c_EU.sh



## Notes

List subdirectories by num files

du -a | cut -d/ -f2 | sort | uniq -c | sort -nr

List subdirectories by size

du -hs * | sort -hr