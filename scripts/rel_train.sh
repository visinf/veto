#!/bin/bash

export OMP_NUM_THREADS=1
export gpu_num=2
export CUDA_VISIBLE_DEVICES="3,4,5,6"

exp_name="veto_x101_fpn"

python -m torch.distributed.launch --master_port 10028 --nproc_per_node=$gpu_num \
       tools/relation_train_net.py \
       --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_VETO.yaml" \
       DEBUG False\
       MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
       EXPERIMENT_NAME "$exp_name" \
       SOLVER.IMS_PER_BATCH $[3*$gpu_num] \
       TEST.IMS_PER_BATCH $[$gpu_num] \
       SOLVER.VAL_PERIOD 2000 \
       SOLVER.CHECKPOINT_PERIOD 2000


