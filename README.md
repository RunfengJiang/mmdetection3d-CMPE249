**Note:** the original README file is named as "README_old.md" from Open-mmlab

### Objective 

Test out MMdetection3d model on Kitti dataset.

### Environment Setting in HPC Deviate from Official Document: https://mmdetection3d.readthedocs.io/
1. Python 3.8.8
2. Cuda 10.1
3. Pytorch 1.6.0 
4. Torchvision 0.7.0

### Specs of Hardware GPU Node
1. Nvidia Tesla P100 12 GB 
2. Driver version 418.87.00
3. CUDA version 10.1

### Kitti 3D Dataset
**Source:** http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d <br>
**Setup:** https://mmdetection3d.readthedocs.io/en/latest/data_preparation.html

### Training 
**Method:** PointPillar 3d 3class <br>
**Terminal execution under the folder of ./mmdetection3d/**
: python tools/train.py configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py --work-dir ./pointpillar_kitti_3d_work_dir/ <br>
**Note:** traning epochs is reduced from 80 to 10, because of the limited job run time in HPC node. 
