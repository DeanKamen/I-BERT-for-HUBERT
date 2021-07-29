#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 python run.py --arch roberta_base --task RTE --restore-file outputs/none/RTE-base/wd0.1_ad0.1_d0.1_lr2e-5/0528-150622_ckpt/checkpoint_best.pt --lr 1e-6
#CUDA_VISIBLE_DEVICES=0 python run.py --arch roberta_base --task SST-2 --restore-file outputs/none/SST-2-base/wd0.1_ad0.1_d0.1_lr1e-5/0601-130538_ckpt/checkpoint_best.pt --lr 1e-6
#CUDA_VISIBLE_DEVICES=0 python run.py --arch roberta_base --task MNLI --restore-file outputs/none/MNLI-base/wd0.1_ad0.1_d0.1_lr1e-5/0528-151504_ckpt/checkpoint_best.pt --lr 1e-6
CUDA_VISIBLE_DEVICES=0 python run.py --arch roberta_base --task QNLI --restore-file outputs/none/QNLI-base/wd0.1_ad0.1_d0.1_lr1e-5/0528-210252_ckpt/checkpoint_best.pt --lr 1e-6
CUDA_VISIBLE_DEVICES=0 python run.py --arch roberta_base --task CoLA --restore-file outputs/none/CoLA-base/wd0.1_ad0.1_d0.1_lr1e-5/0528-223458_ckpt/checkpoint_best.pt --lr 1e-6
CUDA_VISIBLE_DEVICES=0 python run.py --arch roberta_base --task QQP --restore-file outputs/none/QQP-base/wd0.1_ad0.1_d0.1_lr1e-5/0528-170933_ckpt/checkpoint_best.pt --lr 1e-6
CUDA_VISIBLE_DEVICES=0 python run.py --arch roberta_base --task MRPC --restore-file outputs/none/MRPC-base/wd0.1_ad0.1_d0.1_lr1e-5/0601-125337_ckpt/checkpoint_best.pt --lr 1e-6
CUDA_VISIBLE_DEVICES=0 python run.py --arch roberta_base --task STS-B --restore-file outputs/none/STS-B-base/wd0.1_ad0.1_d0.1_lr2e-5/0602-164241_ckpt/checkpoint_best.pt --lr 1e-6