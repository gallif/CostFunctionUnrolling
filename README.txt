################################################################## READ ME ##################################################################
This is the official repository containing CostUnrolling implementation for unsupervised optical flow.
This code mainly consists of the official code of ARFlow (cited in paper) available at: https://github.com/lliuz/ARFlow (Copyright (c) 2020 Liang Liu)
Data is available at the official benchmark websites.
#############################################################################################################################################

# Unsupervised Optical Flow #

# Run the following for model evaluation on the Sintel training set:
python train.py -c configs/sintel_ft_unrolled.json -m checkpoints/sintel_finetuned.pth.tar --n_gpu 2 -e
# Use the following for model evaluation on the KITTI2015 training set:
python train.py -c configs/kitti15_ft_unrolled.json -m checkpoints/kitti15_finetuned.pth.tar --n_gpu 2 -e

# Run the following for finetuning a pretrained model on Sintel:
python train.py -c configs/sintel_ft_unrolled.json --n_gpu 2
# Use the following for finetuning a pretrained model on Sintel:
python train.py -c configs/kitti15_ft_unrolled.json --n_gpu 2

# Run the following for pretraining a new Sintel model:
python train.py -c configs/sintel_raw_unrolled.json --n_gpu 2
# Run the following for pretraining a new KITTI model:
python train.py -c configs/kitti_raw_unrolled.json --n_gpu 2





