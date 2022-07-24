from mmdet3d.apis import init_model, inference_detector, show_result_meshlab

config_file = '/home/015559024/HW_3D_Object_Detection/mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '/home/015559024/HW_3D_Object_Detection/mmdetection3d/pointpillar_kitti_3d_work_dir/latest.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# test a single sample
pcd = '/home/015559024/HW_3D_Object_Detection/mmdetection3d/demo/data/kitti/kitti_000008.bin'
result, data = inference_detector(model, pcd)

# show the results
out_dir = "./"
show_result_meshlab(data, result, out_dir)