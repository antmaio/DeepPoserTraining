#External
import pyrender
from pyrender import Viewer
from psbody.mesh import Mesh
import sys
import logging
import numpy as np
import torch
import os
import xml.etree.ElementTree as ET
import trimesh
from PIL import Image, ImageDraw
from typing import Tuple, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import cv2
#Internal 
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools.vis_tools import colors
#from .yolo_data_gen import extract_from_xml

class CheckerBoard:
    
    def __init__(self, white=(247, 246, 244), black=(146, 163, 171)):
        self.white = np.array(white)/255.
        self.black = np.array(black)/255.
        self.verts, self.faces, self.texts = None, None, None
        self.offset = None

    def init_checker(self, offset, plane='xz', xlength=200000, ylength=50000, square_size=0.5):
        "generate checkerboard and prepare v, f, t"
        checker = self.gen_checker_xy(self.black, self.white, square_size, xlength, ylength)
        rot = np.eye(3)
        if plane == 'xz':
            # rotate around x-axis by 90
            rot[1, 1] = rot[2, 2] = 0
            rot[1, 2] = -1
            rot[2, 1] = 1
        elif plane == 'yz':
            raise NotImplemented
        checker.v = np.matmul(checker.v, rot.T)

        # apply offsets
        checker.v += offset
        self.offset = offset

        self.verts, self.faces, self.texts = self.prep_checker_rend(checker)

    def get_rends(self):
        return self.verts, self.faces, self.texts

    def append_checker(self, checker):
        "append another checker"
        v, f, t = checker.get_rends()
        nv = self.verts.shape[1]
        self.verts = torch.cat([self.verts, v], 1)
        self.faces = torch.cat([self.faces, f+nv], 1)
        self.texts = torch.cat([self.texts, t], 1)

    @staticmethod
    def gen_checkerboard(square_size=0.5, total_size=50.0, plane='xz'):
        "plane: the checkboard is in parallal to which plane"
        checker = CheckerBoard.gen_checker_xy(square_size, total_size)
        rot = np.eye(3)
        if plane == 'xz':
            # rotate around x-axis by 90, so that the checker plane is perpendicular to y-axis
            rot[1, 1] = rot[2, 2] = 0
            rot[1, 2] = -1
            rot[2, 1] = 1
        elif plane == 'yz':
            raise NotImplemented
        checker.v = np.matmul(checker.v, rot.R)
        return checker

    def prep_checker_rend(self, checker:Mesh):
        verts = torch.from_numpy(checker.v.astype(np.float32)).cuda().unsqueeze(0)
        faces = torch.from_numpy(checker.f.astype(int)).cuda().unsqueeze(0)
        nf = checker.f.shape[0]
        texts = torch.zeros(1, nf, 4, 4, 4, 3).cuda()
        for i in range(nf):
            texts[0, i, :, :, :, :] = torch.tensor(checker.fc[i], dtype=torch.float32).cuda()
        return verts, faces, texts

    @staticmethod
    def gen_checker_xy(black, white, square_size=0.5, xlength=500.0, ylength=500.0):
        """
        generate a checker board in parallel to x-y plane
        starting from (0, 0) to (xlength, ylength), in meters
        return: psbody.Mesh
        """
        xsquares = int(xlength / square_size)
        ysquares = int(ylength / square_size)
        verts, faces, texts = [], [], []
        fcount = 0

        for i in range(xsquares):
            for j in range(ysquares):
                p1 = np.array([i * square_size, j * square_size, 0])
                p2 = np.array([(i + 1) * square_size, j * square_size, 0])
                p3 = np.array([(i + 1) * square_size, (j + 1) * square_size, 0])

                verts.extend([p1, p2, p3])
                faces.append([fcount * 3, fcount * 3 + 1, fcount * 3 + 2])
                fcount += 1

                p1 = np.array([i * square_size, j * square_size, 0])
                p2 = np.array([(i + 1) * square_size, (j + 1) * square_size, 0])
                p3 = np.array([i * square_size, (j + 1) * square_size, 0])

                verts.extend([p1, p2, p3])
                faces.append([fcount * 3, fcount * 3 + 1, fcount * 3 + 2])
                fcount += 1

                if (i + j) % 2 == 0:
                    texts.append(black)
                    texts.append(black)
                else:
                    texts.append(white)
                    texts.append(white)

        # now compose as mesh
        mesh = Mesh(v=np.array(verts), f=np.array(faces), fc=np.array(texts))
        mesh.v += np.array([-5, -5, 0])
        return mesh

    @staticmethod
    def from_meshes(meshes, yaxis_up=True, xlength=50, ylength=20):
        """
        initialize checkerboard ground from meshes
        """
        vertices = [x.v for x in meshes]
        if yaxis_up:
            # take ymin
            y_off = np.min(np.concatenate(vertices, 0), 0)
        else:
            # take ymax
            y_off = np.min(np.concatenate(vertices, 0), 0)
        offset = np.array([xlength/2, y_off[1], ylength/2]) # center to origin
        checker = CheckerBoard()
        checker.init_checker(offset, xlength=xlength, ylength=ylength)
        return checker

    @staticmethod
    def from_verts(verts, yaxis_up=True, xlength=5, ylength=5, square_size=0.2):
        """
        verts: (1, N, 3)
        """
        if yaxis_up:
            y_off = torch.min(verts[0], 0)[0].cpu().numpy()
        else:
            y_off = torch.max(verts[0], 0)[0].cpu().numpy()
        # print(verts.shape, y_off.shape)
        offset = np.array([-xlength/2, y_off[1], -ylength/2])
        print(offset, torch.min(verts[0], 0)[0].cpu().numpy(), torch.max(verts[0], 0)[0].cpu().numpy())
        checker = CheckerBoard()
        checker.init_checker(offset, xlength=xlength, ylength=ylength, square_size=square_size)
        return checker

class MeshViewer2(MeshViewer):

    def __init__(self, width=1200, height=800, intrinsic=np.array([[1, 0, 0], [0, 1, 0]]), use_offscreen=True):
        # super().__init__()

        self.width, self.height = width, height
        self.use_offscreen = use_offscreen
        self.render_wireframe = False

        self.mat_constructor = pyrender.MetallicRoughnessMaterial
        self.trimesh_to_pymesh = pyrender.Mesh.from_trimesh

        self.scene = pyrender.Scene(bg_color=colors['white'], ambient_light=(0.3, 0.3, 0.3))

        pc = pyrender.IntrinsicsCamera(intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2])
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 3.0])
        self.camera_node = self.scene.add(pc, pose=camera_pose, name='pc-camera')

        self.figsize = (width, height)

        if self.use_offscreen:
            self.viewer = pyrender.OffscreenRenderer(*self.figsize)
            self.use_raymond_lighting(4.)
        else:
            self.viewer = Viewer(self.scene, use_raymond_lighting=True, viewport_size=self.figsize, cull_faces=False,
                                 run_in_thread=True)

    def updateCam(self, camera_pose, intrinsic):
        self.camera_node.camera.fx = intrinsic[0, 0]
        self.camera_node.camera.fy = intrinsic[1, 1]
        self.camera_node.camera.cx = intrinsic[0, 2]
        self.camera_node.camera.cy = intrinsic[1, 2]
        self.scene.set_pose(self.camera_node, pose=camera_pose)
        
    def get_camera(self):
        return self.camera_node.camera

    def get_projection_matrix(self):
        return self.camera_node.camera.get_projection_matrix(width=self.width, height=self.height)

    def remove_mesh(self):
        for node in self.scene.get_nodes():
            if node.name is not None and 'mesh' in node.name:
                self.scene.remove_node(node)


# utils    
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

#plots 
def init_mesh_viewer(camera_path:str) -> MeshViewer2:
    K, _, (width, height) = extract_from_xml(os.path.join(camera_path, 'Camera_1.xml'))
    #K, _, (width, height) = extract_from_xml(f'{camera_path}camera0_infos_640.xml')
    mv = MeshViewer2(width=width, height=height, intrinsic=K, use_offscreen=True)

    return mv

def render(mv: MeshViewer2, verts: np.ndarray, faces: np.ndarray, savepath: str = 'temp.png', **kwargs) -> None:
    body_mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_colors=np.tile(colors['grey'], (verts.shape[0], 1))
    )

    meshes = [body_mesh]
    mv.set_dynamic_meshes(meshes)

    # Render image
    body_image = mv.render(render_wireframe=False)
    img = Image.fromarray(body_image)
    
    # Overlay 2D projected joints if camera info is provided
    try:
        print('Projection of 3D ground truth in image plane...')
        keypoints = kwargs['keypoints']
        camera_pose = kwargs['camera_pose']
        K = kwargs['K']
        size = kwargs['size']


        assert keypoints is not None, "Projection impossible, check keypoints"
        assert camera_pose is not None, "Projection impossible, check camera_pose"
        assert K is not None, "Projection impossible, check K"
        assert size is not None, "Projection impossible, check size"

        keypoints_2d = world2im(keypoints, camera_pose, K, size)
        
        draw = ImageDraw.Draw(img)
        for x, y in keypoints_2d:
            draw.ellipse((x-3, y-3, x+3, y+3), fill=(255, 0, 0))

    except KeyError:
        pass

    img.save(savepath)

def plot3d(positions_gt: np.ndarray, positions_triang: np.ndarray):
    """
    Scatter plots 3D joint positions for ground truth and triangulated results.

    Parameters
    ----------
    positions_gt : (N, 3) ndarray
        Ground truth 3D positions of joints.
    positions_triang : (N, 3) ndarray
        Triangulated 3D positions of joints.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plots
    ax.scatter(positions_gt[:, 0], positions_gt[:, 1], positions_gt[:, 2], c='blue', label='Ground Truth', s=40)
    ax.scatter(positions_triang[:, 0], positions_triang[:, 1], positions_triang[:, 2], c='red', label='Triangulated', s=40)

    # Fix axis limits based on combined data
    all_points = np.vstack([positions_gt, positions_triang])
    min_bound = np.min(all_points, axis=0)
    max_bound = np.max(all_points, axis=0)

    # Add a margin for clarity
    margin = 0.1 * (max_bound - min_bound)
    ax.set_xlim(min_bound[0] - margin[0], max_bound[0] + margin[0])
    ax.set_ylim(min_bound[1] - margin[1], max_bound[1] + margin[1])
    ax.set_zlim(min_bound[2] - margin[2], max_bound[2] + margin[2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('3D Joint Positions: Ground Truth (Blue) vs Triangulated (Red)')
    plt.savefig("triangulation.png")

# computer vision utils
def get_projection_matrix(K:np.array, width:int, height:int) -> np.array:
    """
    Computes an OpenGL-style 4x4 perspective projection matrix from camera intrinsics.

    This projection matrix maps camera-space 3D points into clip space coordinates
    in [-1, 1]^3 for OpenGL rendering. It incorporates the intrinsic parameters
    (fx, fy, cx, cy) and the image size to match the camera's field of view and
    principal point.

    Parameters
    ----------
    K : (3,3) ndarray
        Camera intrinsic matrix with focal lengths and principal point.
    width : int
        Image width in pixels (viewport width).
    height : int
        Image height in pixels (viewport height).

    Returns
    -------
    P : (4,4) ndarray
        Perspective projection matrix in OpenGL convention.
    """
    width = float(width)
    height = float(height)

    cx, cy = K[0,2], K[1,2]
    fx, fy = K[0,0], K[1,1]
    if sys.platform == 'darwin':
        cx = self.cx * 2.0
        cy = self.cy * 2.0
        fx = self.fx * 2.0
        fy = self.fy * 2.0

    P = np.zeros((4,4))
    P[0][0] = 2.0 * fx / width
    P[1][1] = 2.0 * fy / height
    P[0][2] = 1.0 - 2.0 * cx / width
    P[1][2] = 2.0 * cy / height - 1.0
    P[3][2] = -1.0

    n = 0.05 #default value of znear
    f = 100.0 #default value of znear
    if f is None:
        P[2][2] = -1.0
        P[2][3] = -2.0 * n
    else:
        P[2][2] = (f + n) / (n - f)
        P[2][3] = (2 * f * n) / (n - f)

    return P
    
def world2im(
    keypoints: np.ndarray,
    camera_pose: np.ndarray,
    K: np.ndarray,
    image_size: Tuple[int, int]
) -> np.ndarray:
    """
    Projects 3D keypoints from world space into 2D image pixel coordinates
    using camera extrinsics and intrinsics.

    This function mimics the OpenGL rendering pipeline as used by pyrender:
    - It converts 3D world coordinates to camera coordinates using the camera pose.
    - It applies an OpenGL-style projection matrix derived from intrinsics.
    - It performs perspective division to obtain normalized device coordinates (NDC).
    - It maps NDC [-1, 1] to 2D pixel coordinates, accounting for viewport size and
      origin conventions (OpenGL bottom-left vs image top-left).

    Parameters
    ----------
    keypoints : (N,3) ndarray
        3D points in world coordinates.
    camera_pose : (4,4) ndarray
        Camera-to-world transformation matrix.
    K : (3,3) ndarray
        Camera intrinsic matrix.
    image_size : Tuple[int, int]
        Image size as (width, height).

    Returns
    -------
    projected_points : (M,2) ndarray
        2D image pixel coordinates (origin top-left) of keypoints in front of the camera.
        Only points with valid depth are returned (M ≤ N).
    """
    width, height = image_size

    # Homogeneous world points
    ones = np.ones((keypoints.shape[0], 1))
    keypoints_h = np.hstack([keypoints, ones])  # Nx4

    # Transform world → camera (pyrender node pose)
    world2cam = np.linalg.inv(camera_pose)
    keypoints_cam = (world2cam @ keypoints_h.T).T

    # Project to clip space
    projection_matrix = get_projection_matrix(K, width, height)
    keypoints_clip = (projection_matrix @ keypoints_cam.T).T

    # Perspective divide
    keypoints_ndc = keypoints_clip[:, :3] / keypoints_clip[:, 3:4]

    # Keep only points in front
    in_front = keypoints_ndc[:, 2] > -1  # OpenGL NDC z in [-1,1], closer = -1
    keypoints_ndc = keypoints_ndc[in_front]

    if not len(keypoints_ndc):
        return np.zeros((0, 2))

    # Map from NDC [-1,1] to image pixels
    x_img = (keypoints_ndc[:, 0] + 1) * 0.5 * width
    y_img = (1 - (keypoints_ndc[:, 1] + 1) * 0.5) * height

    return np.stack([x_img, y_img], axis=1)

def triangulate(
    Ks:np.ndarray, 
    camera_poses:np.ndarray, 
    image_sizes:Tuple[int, int], 
    yolo_keypoints:np.ndarray,
    mv:MeshViewer2
):
    
    """
    Triangulate a 3D point from 2D keypoints observed by multiple cameras.

    Parameters
    ----------
    Ks : np.ndarray
        Array of camera intrinsic matrices with shape (n_cams, 3, 3)
    camera_poses : np.ndarray
        Array of camera extrinsic matrices with shape (n_cams, 4, 4)
    image_sizes : List[Tuple[int, int]]
        List of (width, height) tuples for each camera
    yolo_keypoints : torch.Tensor
        Dictionary-like object containing 2D keypoints for each camera
        with keys like 'vcam0' and values of shape (nframes, njoints, 2)

    Returns
    -------
    np.ndarray
        Triangulated 3D point as a numpy array of shape (3,)

    Notes
    -----
    - Currently only supports triangulation from exactly 2 camera views
    - Normalizes 2D points to [0,1] range using image dimensions
    - Uses OpenCV's triangulatePoints function
    """

    """ Method 1 """
    '''
    proj_matrices, points2d = [], []

    for cam_id, cam_key in enumerate(cam_keys_for_triang):
        image_width, image_height = image_sizes[cam_id]
        point2d = yolo_keypoints[cam_key][frame_id, joint_id].cpu().numpy()
        point2d_normalized = np.array([
            point2d[0] / image_width,
            point2d[1] / image_height
        ], dtype=np.float32)
        points2d.append(point2d_normalized)

        #Get projection matrix (K @ [R|t])
        K = Ks[cam_id]
        R = camera_poses[cam_id][:3,:3]
        t = camera_poses[cam_id][:3, 3]
        P = K @ np.hstack((R, t.reshape(3, 1)))
        proj_matrices.append(P)

    #Triangulate
    points2d = np.array(points2d, dtype=np.float32).T
    points3d_hom = cv2.triangulatePoints(
        proj_matrices[0], proj_matrices[1],
        points2d[:, 0:1], points2d[:, 1:2]
    )

    points3d = points3d_hom[:3] / points3d_hom[3].flatten()
    '''
    points3d_list = []
    image_width, image_height = image_sizes

    for joint_id in range(yolo_keypoints.shape[1]):
    
        KMats  = []
        points2d = []

        for cam_id in range(2):
            points2d.append(yolo_keypoints[cam_id,joint_id])
        
            #get camera parameters
            camera_pose = camera_poses[cam_id]
            K = Ks[cam_id]

            mv.updateCam(camera_pose, K)

            KMats.append(mv.viewer._renderer._get_camera_matrices(mv.scene))
            #proj_matrices.append(mv.get_camera().get_projection_matrix(image_width, image_height))

        #from homogeneous to 3D

        mat0, mat1 = (KMats[0][1] @ KMats[0][0])[:3,:], (KMats[1][1] @ KMats[1][0])[:3,:]
        #pmat0, pmat1 = proj_matrices[0][:3], proj_matrices[1][:3]

        kp0, kp1 = np.transpose(copy.copy(points2d[0])), np.transpose(copy.copy(points2d[1]))
        # keypoints at cam 0
        kp0[0] = (kp0[0] - (image_width/2)) / (image_width/2)
        kp0[1] = ((image_height / 2) - kp0[1]) / (image_height/2)
        # keypoints at cam 1
        kp1[0] = (kp1[0] - (image_width/2)) / (image_width/2)
        kp1[1] = ((image_height / 2) - kp1[1]) / (image_height/2)

        #triangulation
        points3d = cv2.triangulatePoints(mat0,mat1,kp0,kp1)
        points3d = points3d[:3] / points3d[3] #homogeneous to heterogeneous

        points3d_list.append(points3d.squeeze())

    return points3d_list



"""
def get_camera_matrices_from_scene(scene, image_width:int, image_height:int):
    main_camera_node = scene.main_camera_node
    if main_camera_node is None:
        raise ValueError('Cannot render scene without a camera')
    projection_matrix = get_projection_matrix(width=image_width, height=image_height)
    pose = scene.get_pose(main_camera_node)
    V = np.linalg.inv(pose)  # V maps from world to camera
    return V, projection_matrix 
"""