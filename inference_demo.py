# from mmdet3d.apis import init_model, inference_detector, show_result_meshlab
# import mmdet3d.core.visualizer.show_result as sr

# config_file = './configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py'
# # download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = './pointpillar_kitti_3d_work_dir/latest.pth'

# # build the model from a config file and a checkpoint file
# model = init_model(config_file, checkpoint_file, device='cuda:0')

# # test a single sample
# pcd = './demo/data/kitti/kitti_000008.bin'
# result, data = inference_detector(model, pcd)
# print(data.keys())
# # show the results
# out_dir = "./"
# show_result_meshlab(data, result, out_dir)

# points = "./show_results"

# sr.show_result()
"""
Convert results into format that is directly readable for meshlab.

    Args:
        points (np.ndarray): Points.
        gt_bboxes (np.ndarray): Ground truth boxes.
        pred_bboxes (np.ndarray): Predicted boxes.
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        show (bool, optional): Visualize the results online. Defaults to False.
        snapshot (bool, optional): Whether to save the online results.
            Defaults to False.
        pred_labels (np.ndarray, optional): Predicted labels of boxes.
            Defaults to None.
"""



