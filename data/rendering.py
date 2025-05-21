#External
import pyrender
from pyrender import Viewer
from psbody.mesh import Mesh
import numpy as np
import torch
import os
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

def init_mesh_viewer(camera_path:str) -> MeshViewer2:
    from .yolo_data_gen import extract_from_xml
    K, _, (width, height) = extract_from_xml(os.path.join(camera_path, 'Camera_1.xml'))
    #K, _, (width, height) = extract_from_xml(f'{camera_path}camera0_infos_640.xml')
    mv = MeshViewer2(width=width, height=height, intrinsic=K, use_offscreen=True)

    return mv