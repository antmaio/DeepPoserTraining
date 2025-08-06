#External
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mim.commands import download
from PIL import Image
import numpy as np
import os 
import matplotlib.pyplot as plt
#Internal
from data.data_config import YOLO_PARENTS, YoloJoints

# config for mmpose
_MODEL_STR_ = 'td-hm_hrnet-w32_8xb64-210e_coco-384x288' # change to desired model 
#_MODEL_STR_ = 'rtmo-t_8xb32-600e_body7-416x416' # change to desired model 

if 'coco' in _MODEL_STR_:
    KEYPOINTS_TOPOLOGY = YoloJoints
else:
    raise NotImplementedError
#_CONFIG_FILE_ = f'{_MODEL_STR_}.py'
#_CHECKPOINT_FILE_ = 'td-hm_hrnet-w32_8xb64-210e_coco-384x288-ca5956af_20220909.pth'

#config_file = os.path.join(_MM_MODEL_PATH_, _CONFIG_PATH_, _CONFIG_FILE_)
#checkpoint_file = os.path.join(_MM_MODEL_PATH_, _CHECKPOINT_PATH_, _CHECKPOINT_FILE_)

def visualize_mmpose_results(results, body_image, mm_pose_model)->None:

    """
    keypoints = results[0].pred_instances.keypoints.squeeze()
    njoints, _ = keypoints.shape

    #get edges
    def get_yolo_edges(parents):
        edges = []
        for child, parent in enumerate(parents):
            if parent != -1:
                edges.append((parent, child))
        return edges
    
    def plot_pose2d_joints(pose2d, edges, title='2D Pose', figsize=(6, 8)):
        plt.figure(figsize=figsize)
        plt.scatter(pose2d[:, 0], pose2d[:, 1], color='red', zorder=2)
        for start, end in edges:
            x = [pose2d[start, 0], pose2d[end, 0]]
            y = [pose2d[start, 1], pose2d[end, 1]]
            plt.plot(x, y, color='blue', linewidth=2, zorder=1)
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.title(title)
        plt.savefig("output_mmpose.jpeg")

    
    yolo_edges = get_yolo_edges(YOLO_PARENTS)
    plot_pose2d_joints(keypoints, yolo_edges)
    """
    
    results = merge_data_samples(results)
    visualizer = VISUALIZERS.build(mm_pose_model.cfg.visualizer)
    visualizer.set_dataset_meta(
        mm_pose_model.dataset_meta, skeleton_style='mmpose')

    visualizer.add_datasample(
        'result',
        body_image,
        data_sample=results,
        draw_gt=False,
        draw_bbox=True,
        draw_heatmap=True,
        show_kpt_idx=False,
        skeleton_style='mmpose',
        show=False,
        out_file='./output_mmpose.jpeg',
        kpt_thr = 0
    )
    

def download_config(config: str, dest: str = '.'):

    download(
        package='mmpose',
        configs=[config],
        dest_root=dest
    )

if __name__ == '__main__':

    """
    register_all_modules()
    download_config("td-hm_hrnet-w32_8xb64-210e_coco-384x288")
    """

    """
    config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
    checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

    model = init_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

    # please prepare an image with person
    #TODO pass np array instead of str
    img = np.array(Image.open('demo.jpg'))

    results = inference_topdown(model, img)
    pred_instances = results[0].pred_instances
    print(pred_instances.keypoints)
    """
    pass
