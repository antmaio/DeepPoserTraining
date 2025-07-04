#External
from PIL import Image
import os
import ultralytics
from ultralytics import YOLO
import torch
import numpy as np
import trimesh
import glob
#Internal 
from debug.rendering import MeshViewer2
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from body_visualizer.tools.vis_tools import colors
from debug.rendering import extract_from_xml

def visualize_yolo_results(res, cam:str)->None:
    annotated_frame = res[0].plot()
    im = Image.fromarray(annotated_frame)
    os.makedirs('processed_images', exist_ok=True)
    im.save(os.path.join('processed_images', f"yolo_results_on_{cam}.jpeg"))

def init_yolo(yolo_model:str='yolov8x-pose.pt'):
    # Create yolo model
    model = YOLO(yolo_model)
    model.to(0)
    # Perform object detection on an image using the model
    print('cuda:', torch.cuda.is_available())

    return model

def run_yolo(
        yolo_model:ultralytics.models.yolo.model.YOLO, 
        mv:MeshViewer2, 
        bm:BodyModel, 
        body_pose_world, 
        nb_frames:int,
        frame_to_render:int,
        camera_files
    ):

    # Create the body model
    faces = c2c(bm.f)

    cam_2d = []
    print(f'Processing {nb_frames} frames ...')

    keyPoints, _, _, _ = get_keypoints(
        mv=mv, 
        model=yolo_model, 
        body_pose_hand=body_pose_world,
        faces=faces,
        camera_files=camera_files,
        frame_to_render=frame_to_render
    )

    # Store results for each camera
    for cam_idx in range(len(keyPoints)):
        # Ensure we have enough lists to store each camera's data
        if len(cam_2d) <= cam_idx:
            cam_2d.append([])
        
        # Extract 2D coordinates (x,y) and store
        cam_2d[cam_idx].append(keyPoints[cam_idx][0][:, 0:2])

    cam_2d_arrays = [np.array(cam_data) for cam_data in cam_2d]

    return cam_2d_arrays

def get_keypoints(mv:MeshViewer2, frame_to_render:int, model, body_pose_hand, faces, camera_files):
    ret = []
    confs = []
    KMatCont = []
    PMatCont = []

    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_hand.v[frame_to_render]), faces=faces,
                                vertex_colors=np.tile(colors['grey'], (6890, 1)))

    mv.set_dynamic_meshes([body_mesh])

    # Get all camera XML files (any naming pattern)
    for camera_file in camera_files:
        K, camera_pose, _ = extract_from_xml(camera_file)
        camera_pose = np.vstack((camera_pose, [0, 0, 0, 1]))

        mv.updateCam(camera_pose, K)

        results, confidences, KMat, PMat = get_keypoints_by_cam(mv=mv,model=model)

        KMatCont.append(KMat)
        PMatCont.append(PMat)

        ret.append(results[0])
        confs.append(confidences[0])

    return ret, confs, KMatCont, PMatCont

def get_keypoints_by_cam(mv, model):

    KMat = mv.viewer._renderer._get_camera_matrices(mv.scene)
    PMat = mv.get_projection_matrix()
    body_image = mv.render(render_wireframe=False)

    results = model.predict(body_image, device=0, verbose=False, stream=True)

    ret = []
    conf = []
    for res in results:
        # Save yolo image
        
        ret.append(res[0].keypoints.data.cpu().numpy())
        conf.append(res[0].keypoints.conf.cpu().numpy())
        #visualize_yolo_results(res, cam, fId)


    return ret, conf, KMat, PMat
