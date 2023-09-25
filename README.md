# Vision Relation Transformer for Unbiased Scene Graph Generation (ICCV 2023) 

This is the official repository for the paper ["Vision Relation Transformer for Unbiased Scene Graph Generation"](https://arxiv.org/abs/2308.09472).

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.


## Pretrained Models

For VG dataset, the pretrained object detector we used is provided by [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), you can download it from [this link](https://1drv.ms/u/s!AjK8-t5JiDT1kxT9s3JwIpoGz4cA?e=usU6TR). For GQA dataset, we used the pretrained object detector provided by SHA-GCL-for-SGG which can be downloaded from [this link](https://1drv.ms/u/s!AjK8-t5JiDT1kxBfihou2smfXFV9?e=VtyoR7). Modify the pretrained weight parameter `MODEL.PRETRAINED_DETECTOR_CKPT` in configs yaml `configs/VETO_final.yaml` to the path of corresponding pretrained rcnn weight to make sure you load the detection weight parameter correctly.


### Scene Graph Generation Model
You can follow the following instructions to train your own, which takes 1 GPU to train each SGG model. The results should be very close to the reported results given in paper.

Following script trains VETO vanilla for PredCls (For SGCls set MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False, For SGDet set MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False)
```
       python ./tools/relation_train_net.py --config-file 
       "configs/VETO_final.yaml"
       MODEL.ROI_RELATION_HEAD.PREDICTOR VETOPredictor 
       GLOBAL_SETTING.DATASET_CHOICE 'VG' MODEL.ROI_RELATION_HEAD.USE_GT_BOX True 
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True 
       GLOBAL_SETTING.BETA_LOSS False
       SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 1 
       SOLVER.MAX_ITER 125000 SOLVER.VAL_PERIOD 5000 
       SOLVER.CHECKPOINT_PERIOD 5000 DEBUG False 
       SOLVER.PRE_VAL False ENSEMBLE_LEARNING.ENABLED False
       EXPERIMENT_NAME "VG_VETO_vanilla"

```
Following script trains VETO + Rwt for PredCls
```
       python ./tools/relation_train_net.py --config-file 
       "configs/VETO_final.yaml" 
       MODEL.ROI_RELATION_HEAD.PREDICTOR VETOPredictor
       GLOBAL_SETTING.DATASET_CHOICE 'VG' MODEL.ROI_RELATION_HEAD.USE_GT_BOX True 
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True 
       GLOBAL_SETTING.BETA_LOSS True
       SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 1 
       SOLVER.MAX_ITER 125000 SOLVER.VAL_PERIOD 5000 
       SOLVER.CHECKPOINT_PERIOD 5000 DEBUG False 
       SOLVER.PRE_VAL False ENSEMBLE_LEARNING.ENABLED False
       EXPERIMENT_NAME "VG_VETO_beta"

```

Following script trains VETO + MEET for PredCls
```
       python ./tools/relation_train_net.py --config-file 
       "configs/VETO_final.yaml" 
       MODEL.ROI_RELATION_HEAD.PREDICTOR VETOPredictor_MEET
       GLOBAL_SETTING.DATASET_CHOICE 'VG' MODEL.ROI_RELATION_HEAD.USE_GT_BOX True 
       MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True 
       GLOBAL_SETTING.BETA_LOSS True
       SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 1 
       SOLVER.MAX_ITER 125000 SOLVER.VAL_PERIOD 5000 
       SOLVER.CHECKPOINT_PERIOD 5000 DEBUG False 
       SOLVER.PRE_VAL False ENSEMBLE_LEARNING.ENABLED True
       EXPERIMENT_NAME "VG_VETO_MEET"

```

## Test
By replacing the parameter of `MODEL.WEIGHT` to the trained model weight and selected dataset name in `DATASETS.TEST`, you can directly eval the model on validation or test set.

## Cite
```
       @inproceedings{sudhakaran2023vision,
         title={Vision Relation Transformer for Unbiased Scene Graph Generation},
         author={Sudhakaran, Gopika and Dhami, Devendra Singh and Kersting, Kristian and Roth, Stefan},
         booktitle = {2023 {IEEE/CVF} International Conference on Computer Vision (ICCV), Paris, France, October 2-6, 2023},
         year      = {2023},
         publisher = {{IEEE}}, 
         pages     = ....
       }
```
## Acknowledgment
This repository is developed on top of the following code bases:
1. Scene graph benchmarking framework develped by [KaihuaTang](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
2. A Toolkit for Scene Graph Benchmark in Pytorch by [Rongjie Li](https://github.com/SHTUPLUS/PySGG)
3. Stacked Hybrid-Attention and Group Collaborative Learning for Unbiased Scene Graph Generation in Pytorch by [Xingning Dong](Stacked Hybrid-Attention and Group Collaborative Learning for Unbiased Scene Graph Generation)
