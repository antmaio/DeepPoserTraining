import numpy as np
import torch
import xml.etree.ElementTree as ET
import glob
import os
import numpy as np
import trimesh
import ultralytics
import re
from PIL import Image

#Internal
from body_visualizer.tools.vis_tools import colors
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from data.rendering import CheckerBoard,MeshViewer2
from data.data_config import YoloJoints
from human_body_prior.body_model.body_model import BodyModel
from data.utils_yolo import visualize_yolo_results
#from data.utils_mmpose import visualize_mmpose_results


import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

'''
""" Projection """
def to_homogeneous(points_world):
    """
    Convert 3D points to homogeneous coordinates.
    """
    assert len(points_world.shape) == 3
    assert points_world.shape[-1] == 3
    nframes, njoints, _ = points_world.shape
    ones = np.ones((nframes, njoints, 1))  # Create a column of ones
    points_world_h = np.concatenate((points_world, ones), axis=-1)  # Concatenate along the last dimension
    return points_world_h

def to_heterogeneous(points_world_h):
    """
    Convert points from homogeneous to non-homogeneous coordinates.
    """
    assert len(points_world_h.shape) == 3
    assert points_world_h.shape[-1] == 4
    return points_world_h[..., :3]  # Remove the homogeneous coordinate

def world_to_cam(data3d, ex_params):
    """
    Project world coordinates to camera coordinates.
    """
    data3d_h = to_homogeneous(data3d)  # Convert to homogeneous coordinates
    # Compute the inverse
    ex_params = np.linalg.inv(ex_params)
    CORRECTION_MATRIX = np.array(
        [[1,0,0,0],
        [0,-1,0,0],
        [0,0,-1,0],
        [0,0,0,1]]
    )
    ex_params = np.dot(CORRECTION_MATRIX, ex_params)
    data3d_cam_h = np.einsum('ij,nkj->nki', ex_params, data3d_h)  # Matrix multiplication for each joint
    data3d_cam = to_heterogeneous(data3d_cam_h)  # Remove the homogeneous coordinate
    return data3d_cam

def cam_to_image(data3d_cam, intr_params):
    """
    Project 3D camera coordinates to 2D image plane.
    """
    # Unpack the camera coordinates
    X_camera = data3d_cam[..., 0]
    Y_camera = data3d_cam[..., 1]
    Z_camera = data3d_cam[..., 2]

    # Normalize by depth (Z)
    x_normalized = X_camera / Z_camera  # Shape: (nframes, njoints)
    y_normalized = Y_camera / Z_camera  # Shape: (nframes, njoints)

    # Stack normalized coordinates into homogeneous 2D points
    points_normalized = np.stack((x_normalized, y_normalized, np.ones_like(x_normalized)), axis=-1)  # Shape: (nframes, njoints, 3)

    # Project into the image plane using the intrinsic matrix
    points_image_h = np.einsum('ij,nkj->nki', intr_params, points_normalized)  # Matrix multiplication

    # Extract pixel coordinates (u, v) from homogeneous coordinates
    u = points_image_h[..., 0]
    v = points_image_h[..., 1]

    # Combine u and v into a final array of shape (nframes, njoints, 2)
    points_image = np.stack((u, v), axis=-1)
    return points_image

def projection(data3d, ex_params, intr_params):
    data3d_cam = world_to_cam(data3d, ex_params)
    data2d = cam_to_image(data3d_cam, intr_params)
    return data2d
'''

""" Rendering Debug """
def generate_checker_mesh():
    generator = CheckerBoard()
    checker = generator.gen_checker_xy(generator.black, generator.white)
    checker_mesh = trimesh.Trimesh(checker.v,checker.f,process=False,face_colors=checker.fc)
    checker_mesh.apply_translation([0, 0, 0])
    return checker_mesh

def generate_camera_mesh(camera_path:str, camera_id:int, apply_transform:bool=False):
    K, camera_pose, _ = extract_from_xml(f'{camera_path}Camera_{camera_id}.xml')
    camera_pose = np.vstack((camera_pose, [0, 0, 0, 1]))

    # Main camera body
    camera_body = trimesh.creation.box(extents=(0.1, 0.05, 0.05))  # Width, height, depth
    # Create a cylinder for the lens
    lens = trimesh.creation.cylinder(radius=0.05, height=0.08, sections=32)
    lens.apply_translation([0, 0, 0.05])  # Move the lens to the front of the camera body
    # Create a button using a small sphere
    button = trimesh.creation.icosphere(subdivisions=3, radius=0.01)
    button.apply_translation([0.02, 0.04, 0.06])  # Position the button on top of the body
    # Combine all parts into a single mesh
    camera_mesh = trimesh.util.concatenate([camera_body, lens, button])
    camera_mesh.apply_transform(camera_pose)
    # Create the axis system
    axis_length = 0.2  # Length of the axis lines
    camera_axes_mesh= trimesh.creation.axis(axis_length=axis_length)  # Trimesh utility to create an axis object

    camera_axes_mesh.apply_transform(camera_pose)
    return camera_mesh, camera_axes_mesh

def generate_gt3D_joints(global_positions_head_centered):
    """ Visualize ground truth 3D positions """
    sphere_color=(1, 0, 0)  # Red spheres
    spheres = []
    for joint_position in global_positions_head_centered.squeeze(): 
        sphere = trimesh.primitives.Sphere(radius=0.05, center=joint_position)
        # Set sphere color
        sphere.visual.face_colors = np.array(sphere_color)

        # Add to the list
        spheres.append(sphere)
    return spheres

def generate_new_scene_pov(mv, K):
    t_x = 3.5
    t_y = 3.0
    t_x = 9.0
    t_y = 4.0
    t_z = 15.0

    theta_x = 0
    theta_y = 15/2
    theta_z = 45
    # Example camera pose matrix
    theta_x = np.radians(theta_x)  # Convert to radians
    theta_y = np.radians(theta_y)  # Convert to radians
    theta_z = np.radians(theta_z)  # Convert to radians
    # Create the translation vector (reshaped to 3x1 for stacking)
    translation = np.array([[t_x], [t_y], [t_z]])  # Shape: (3, 1)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    R_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])    
    R_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])

    # Combine rotations: First X, then Y, then Z
    pov_pose = np.dot(R_z, np.dot(R_y, R_x))  # Apply Z after Y and X
    # Add translation (concatenate along columns)
    pov_pose = np.hstack((pov_pose, translation))  # Shape: (3, 4)
    pov_pose = np.vstack((pov_pose, [0, 0, 0, 1]))
    
    mv.updateCam(pov_pose, K)
    return mv

def save_body_image(body_image):
    image = Image.fromarray(body_image)
    image.save('body_image_cam_1.png')

""" Pose estimation """
def inference(body_image, cam, frame_path, fId, orig_file, **kwargs):

    yolo_model  = kwargs.get('yolo_model') #pose estimation from ultralytics
    inferencer  = kwargs.get('mm_pose_model') #pose estimation from mmpose
    topology    = kwargs.get('topology')

    ret = []
    conf = []
    #save_body_image(body_image)
    
    # --- Pose Estimation with Yolo model --- 
    if yolo_model:

        results = yolo_model.predict(body_image, device=0, verbose=False, stream=True)

        for res in results:
        
            if len(res) > 0:
                ret.append(res[0].keypoints.data.cpu().numpy())
                conf.append(res[0].keypoints.conf.cpu().numpy())
                visualize_yolo_results(res, cam)

            else:
                save_bad_frame(frame_path, fId, orig_file)
                ret.append(np.zeros((1, topology.NUM_JTS, 2), dtype=float))
                conf.append(np.zeros((1, topology.NUM_JTS), dtype=float))
 
        return ret, conf

    # --- Pose Estimation with model from mmpose --- 
    elif inferencer:

        result_generator = inferencer(body_image, show=False)
        results = next(result_generator)

        if len(results['predictions']) > 0: #if pose on image
            predictions_mmpose = results['predictions'][0][0] # first image - first prediction
            keypoints = np.expand_dims(np.array(predictions_mmpose['keypoints']), axis=0)
            keypoint_scores = np.expand_dims(np.array(predictions_mmpose['keypoint_scores']), axis=0)
            ret.append(keypoints)
            conf.append(keypoint_scores)

        else:
            save_bad_frame(frame_path, fId, orig_file)
            ret.append(np.zeros((1, topology.NUM_JTS, 2), dtype=float))
            conf.append(np.zeros((1, topology.NUM_JTS), dtype=float))

        del result_generator #save memory

        return ret, conf


def get_keypoints_by_cam(mv, cam, fId, frame_path, orig_file, im, **kwargs):

    KMat = mv.viewer._renderer._get_camera_matrices(mv.scene)
    PMat = mv.get_projection_matrix()
    body_image = mv.render(render_wireframe=False)

    ret, conf = inference( 
        body_image  = body_image,
        cam         = cam,
        frame_path  = frame_path,
        fId         = fId,
        orig_file   = orig_file,
        **kwargs    
    )

    return ret, conf, KMat, PMat

def get_keypoints(fId:int, mv:MeshViewer2, body_pose_hand, faces, frame_path, orig_file, camera_path, **kwargs):
    ret = []
    confs = []
    KMatCont = []
    PMatCont = []

    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_hand.v[fId]), faces=faces,
                                vertex_colors=np.tile(colors['grey'], (6890, 1)))

    mv.set_dynamic_meshes([body_mesh])

    # Get all camera XML files (any naming pattern)
    camera_files = glob.glob(os.path.join(camera_path, '*.xml'))
    
    # Ensure sorted list of camera_files
    def get_cam_id(path):
        filename = os.path.basename(path)
        _match = re.search(r'Camera_(\d+)\.xml', filename)
        return int(_match.group(1)) if _match else float('inf')

    camera_files.sort(key=get_cam_id) 

    for camera_file in camera_files:
        K, camera_pose, _ = extract_from_xml(camera_file)
        camera_pose = np.vstack((camera_pose, [0, 0, 0, 1]))
        cam = os.path.basename(camera_file).split('.')[0] 

        mv.updateCam(camera_pose, K)
        im=None

        results, confidences, KMat, PMat = get_keypoints_by_cam(mv=mv, 
            #model=model, 
            cam=cam, 
            fId=fId,
            frame_path=frame_path, 
            orig_file=orig_file, 
            im=im,
            **kwargs
        )

        KMatCont.append(KMat)
        PMatCont.append(PMat)

        ret.append(results[0])
        confs.append(confidences[0])

    return ret, confs, KMatCont, PMatCont

def run_yolo(
        mv:MeshViewer2, 
        bm:BodyModel, 
        body_pose_world, 
        nb_frames:int, 
        orig_file:str, 
        frame_path:str, 
        idx:int,
        camera_path:str,
        **kwargs
    ):

    # Choose the device to run the body model on.
    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(comp_device)

    # Create the body model
    faces = c2c(bm.f)

    # Image resolution
    Confidences, cam_2d = [], []
    print(f'Processing {nb_frames} frames ...')

    for frameId in range(nb_frames + 1):

        #print(f'\r{frameId}/{nb_frames}', end='')

        frame_path_id = f"{frame_path}/{idx}.pkl"

        keyPoints, conf, _, _ = get_keypoints(fId=frameId,
            mv=mv, 
            #model=yolo_model, 
            body_pose_hand=body_pose_world,
            faces=faces, 
            frame_path=frame_path_id, 
            orig_file=orig_file,
            camera_path=camera_path,
            **kwargs
        )

        # Store confidences (assuming same across cameras or needs processing)
        Confidences.append(conf)
        # Store results for each camera
        for cam_idx in range(len(keyPoints)):
            # Ensure we have enough lists to store each camera's data
            if len(cam_2d) <= cam_idx:
                cam_2d.append([])
            
            # Extract 2D coordinates (x,y) and store
            cam_2d[cam_idx].append(keyPoints[cam_idx][0][:, 0:2])

    #print('\n')

    Confidences = np.array(Confidences).squeeze()
    Confidences = np.transpose(Confidences, (0, 2, 1))
    # Convert each camera's list to numpy array
    cam_2d_arrays = [np.array(cam_data) for cam_data in cam_2d]

    return cam_2d_arrays, Confidences

""" Utils functions """    
def extract_from_xml(file_path):
    # Parse the XML file using ElementTree and get the root element
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Find the 'Intrinsics' element in the XML and extract the 'data' text
    intrinsics = root.find('Intrinsics')
    data_text = intrinsics.find('data').text
    # Split the text into lines and convert each line into a list of floats
    K = np.array([list(map(float, line.split())) for line in data_text.strip().split('\n')])

    # Find the 'CameraMatrix' element in the XML and extract the 'data' text
    extrinsics = root.find('CameraMatrix')
    data_text = extrinsics.find('data').text

    # Split the text into lines and convert each line into a list of floats
    camera_pose = np.array([list(map(float, line.split())) for line in data_text.strip().split('\n')])

    #Imae size
    image_width = int(root.find('image_width').text)
    image_height = int(root.find('image_height').text)
    image_size = (image_width, image_height)
    
    return K, camera_pose, image_size

def save_bad_frame(frame_path, fId, orig_file):

    with open(f'./yolo_bad_frames.txt', 'a') as txtfile:
        txtfile.write(f'{orig_file} {frame_path} {fId}\n')
    return