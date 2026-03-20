# External
import os
import numpy as np
import torch
import pyrender
from pyrender import Viewer
from psbody.mesh import Mesh
# Internal
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools.vis_tools import colors


class CheckerBoard:
    """Generates and manages a checkerboard ground-plane mesh for rendering."""

    def __init__(self, white=(247, 246, 244), black=(146, 163, 171)):
        self.white = np.array(white) / 255.0
        self.black = np.array(black) / 255.0
        self.verts = self.faces = self.texts = None
        self.offset = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def init_checker(self, offset, plane='xz', xlength=200_000, ylength=50_000, square_size=0.5):
        """Generate checkerboard geometry and cache vertices / faces / textures."""
        checker = self.gen_checker_xy(self.black, self.white, square_size, xlength, ylength)

        rot = self._plane_rotation(plane)
        checker.v = checker.v @ rot.T
        checker.v += offset
        self.offset = offset

        self.verts, self.faces, self.texts = self._prep_for_render(checker)

    def get_rends(self):
        return self.verts, self.faces, self.texts

    def append_checker(self, other: 'CheckerBoard'):
        """Concatenate another checker's geometry into this one."""
        v, f, t = other.get_rends()
        nv = self.verts.shape[1]
        self.verts = torch.cat([self.verts, v], dim=1)
        self.faces = torch.cat([self.faces, f + nv], dim=1)
        self.texts = torch.cat([self.texts, t], dim=1)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def from_meshes(meshes, yaxis_up=True, xlength=50, ylength=20):
        """Initialise a checkerboard ground plane sized to fit *meshes*."""
        vertices = np.concatenate([m.v for m in meshes], axis=0)
        y_off = np.min(vertices, axis=0)  # works for both yaxis_up and not
        offset = np.array([xlength / 2, y_off[1], ylength / 2])
        checker = CheckerBoard()
        checker.init_checker(offset, xlength=xlength, ylength=ylength)
        return checker

    @staticmethod
    def from_verts(verts, yaxis_up=True, xlength=5, ylength=5, square_size=0.2):
        """
        Initialise from a tensor of vertices with shape (1, N, 3).
        """
        extremum = torch.min if yaxis_up else torch.max
        y_off = extremum(verts[0], dim=0)[0].cpu().numpy()
        offset = np.array([-xlength / 2, y_off[1], -ylength / 2])
        checker = CheckerBoard()
        checker.init_checker(offset, xlength=xlength, ylength=ylength, square_size=square_size)
        return checker

    # ------------------------------------------------------------------
    # Low-level geometry
    # ------------------------------------------------------------------

    @staticmethod
    def gen_checker_xy(black, white, square_size=0.5, xlength=500.0, ylength=500.0) -> Mesh:
        """
        Build a checkerboard mesh in the XY plane.
        Returns a psbody Mesh with per-face colours.
        """
        xsquares = int(xlength / square_size)
        ysquares = int(ylength / square_size)
        verts, faces, face_colors = [], [], []
        fcount = 0

        for i in range(xsquares):
            for j in range(ysquares):
                color = black if (i + j) % 2 == 0 else white
                x0, x1 = i * square_size, (i + 1) * square_size
                y0, y1 = j * square_size, (j + 1) * square_size

                # First triangle
                verts += [
                    [x0, y0, 0], [x1, y0, 0], [x1, y1, 0],
                ]
                faces.append([fcount * 3, fcount * 3 + 1, fcount * 3 + 2])
                face_colors.append(color)
                fcount += 1

                # Second triangle
                verts += [
                    [x0, y0, 0], [x1, y1, 0], [x0, y1, 0],
                ]
                faces.append([fcount * 3, fcount * 3 + 1, fcount * 3 + 2])
                face_colors.append(color)
                fcount += 1

        mesh = Mesh(v=np.array(verts), f=np.array(faces), fc=np.array(face_colors))
        mesh.v += np.array([-5, -5, 0])
        return mesh

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _plane_rotation(plane: str) -> np.ndarray:
        """Return the rotation matrix that maps the XY plane to *plane*."""
        rot = np.eye(3)
        if plane == 'xz':
            rot[1, 1] = rot[2, 2] = 0
            rot[1, 2] = -1
            rot[2, 1] = 1
        elif plane == 'yz':
            raise NotImplementedError("'yz' plane not yet implemented.")
        return rot

    def _prep_for_render(self, checker: Mesh):
        verts = torch.from_numpy(checker.v.astype(np.float32)).cuda().unsqueeze(0)
        faces = torch.from_numpy(checker.f.astype(int)).cuda().unsqueeze(0)
        nf = checker.f.shape[0]
        texts = torch.zeros(1, nf, 4, 4, 4, 3).cuda()
        for i in range(nf):
            texts[0, i] = torch.tensor(checker.fc[i], dtype=torch.float32).cuda()
        return verts, faces, texts


class MeshViewer2(MeshViewer):
    """Extended MeshViewer supporting camera intrinsics updates and mesh removal."""

    def __init__(self, width=1200, height=800,
                 intrinsic=np.array([[1, 0, 0], [0, 1, 0]]),
                 use_offscreen=True):
        self.width = width
        self.height = height
        self.use_offscreen = use_offscreen
        self.render_wireframe = False

        self.mat_constructor = pyrender.MetallicRoughnessMaterial
        self.trimesh_to_pymesh = pyrender.Mesh.from_trimesh

        self.scene = pyrender.Scene(bg_color=colors['white'], ambient_light=(0.3, 0.3, 0.3))

        pc = pyrender.IntrinsicsCamera(
            intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        )
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = [0, 0, 3.0]
        self.camera_node = self.scene.add(pc, pose=camera_pose, name='pc-camera')

        self.figsize = (width, height)

        if self.use_offscreen:
            self.viewer = pyrender.OffscreenRenderer(*self.figsize)
            self.use_raymond_lighting(4.0)
        else:
            self.viewer = Viewer(
                self.scene,
                use_raymond_lighting=True,
                viewport_size=self.figsize,
                cull_faces=False,
                run_in_thread=True,
            )

    def updateCam(self, camera_pose: np.ndarray, intrinsic: np.ndarray):
        cam = self.camera_node.camera
        cam.fx = intrinsic[0, 0]
        cam.fy = intrinsic[1, 1]
        cam.cx = intrinsic[0, 2]
        cam.cy = intrinsic[1, 2]
        self.scene.set_pose(self.camera_node, pose=camera_pose)

    def get_camera(self):
        return self.camera_node.camera

    def get_projection_matrix(self):
        return self.camera_node.camera.get_projection_matrix(
            width=self.width, height=self.height
        )

    def remove_mesh(self):
        for node in self.scene.get_nodes():
            if node.name is not None and 'mesh' in node.name:
                self.scene.remove_node(node)


def init_mesh_viewer(camera_path: str) -> MeshViewer2:
    """Instantiate a MeshViewer2 using camera intrinsics from *camera_path*/Camera_1.xml."""
    from data.yolo_data_gen import extract_from_xml  # local import to avoid circular deps

    K, _, (width, height) = extract_from_xml(os.path.join(camera_path, 'Camera_1.xml'))
    return MeshViewer2(width=width, height=height, intrinsic=K, use_offscreen=True)
