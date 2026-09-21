"""
Scene-rendering utilities: a checkerboard ground plane and an extended
``pyrender``-based mesh viewer.

``CheckerBoard`` builds a textured ground-plane mesh (as GPU tensors ready
for a differentiable/batched renderer) that can be sized to fit a set of
meshes or vertices, used as a visual reference plane in rendered scenes.

``MeshViewer2`` extends ``body_visualizer``'s ``MeshViewer`` with the ability
to update camera intrinsics/extrinsics after construction and to remove
meshes from the scene, which is needed when rendering the same scene from
several different calibrated cameras (see ``data.yolo_data_gen``).
"""

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
        """Initialize the checkerboard's colors.

        Args:
            white: RGB triple (0-255 range) for the "white" squares.
            black: RGB triple (0-255 range) for the "black" squares.
        """
        self.white = np.array(white) / 255.0
        self.black = np.array(black) / 255.0
        self.verts = self.faces = self.texts = None
        self.offset = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def init_checker(self, offset, plane='xz', xlength=200_000, ylength=50_000, square_size=0.5):
        """Generate checkerboard geometry and cache vertices / faces / textures.

        Builds a checkerboard in the XY plane, rotates it into the requested
        plane, translates it by ``offset``, and prepares GPU tensors for
        rendering (stored on ``self.verts``/``self.faces``/``self.texts``).

        Args:
            offset: (3,) array-like world-space translation applied to the
                checkerboard after rotation.
            plane: Which plane the checkerboard should lie in after
                rotation. Currently only ``'xz'`` is implemented; ``'yz'``
                raises ``NotImplementedError``.
            xlength: Extent of the checkerboard along its local X axis,
                before rotation, in world units.
            ylength: Extent of the checkerboard along its local Y axis,
                before rotation, in world units.
            square_size: Side length of each individual checker square.
        """
        checker = self.gen_checker_xy(self.black, self.white, square_size, xlength, ylength)

        rot = self._plane_rotation(plane)
        checker.v = checker.v @ rot.T
        checker.v += offset
        self.offset = offset

        self.verts, self.faces, self.texts = self._prep_for_render(checker)

    def get_rends(self):
        """Return the cached render-ready geometry.

        Returns:
            A tuple ``(verts, faces, texts)`` of GPU tensors as produced by
            ``_prep_for_render``: batched vertices, batched face indices,
            and per-face texture tensors.
        """
        return self.verts, self.faces, self.texts

    def append_checker(self, other: 'CheckerBoard'):
        """Concatenate another checker's geometry into this one.

        Face indices from ``other`` are offset by this checker's current
        vertex count so the combined mesh remains valid.

        Args:
            other: Another initialized ``CheckerBoard`` whose geometry
                (``verts``, ``faces``, ``texts``) will be appended to this
                instance's geometry in place.
        """
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
        """Initialise a checkerboard ground plane sized to fit *meshes*.

        Positions the plane so that it sits at the lowest point of the
        combined vertex set (i.e. just under the meshes), centered under
        them in X/Z.

        Args:
            meshes: Iterable of mesh-like objects exposing a ``.v``
                (N, 3) vertex array (e.g. ``psbody.mesh.Mesh`` instances).
            yaxis_up: Whether Y is the up axis. Currently unused beyond
                documenting intent (the same min-based offset is used
                either way).
            xlength: Extent of the checkerboard along X.
            ylength: Extent of the checkerboard along its (pre-rotation) Y
                axis (i.e. depth once placed in the XZ plane).

        Returns:
            A new, initialized ``CheckerBoard`` instance.
        """
        vertices = np.concatenate([m.v for m in meshes], axis=0)
        y_off = np.min(vertices, axis=0)  # works for both yaxis_up and not
        offset = np.array([xlength / 2, y_off[1], ylength / 2])
        checker = CheckerBoard()
        checker.init_checker(offset, xlength=xlength, ylength=ylength)
        return checker

    @staticmethod
    def from_verts(verts, yaxis_up=True, xlength=5, ylength=5, square_size=0.2):
        """Initialise a checkerboard ground plane sized to fit a vertex tensor.

        Args:
            verts: Tensor of shape (1, N, 3) containing the vertices to fit
                the ground plane under.
            yaxis_up: If True, the plane is placed at the vertices' minimum
                Y (i.e. below them, assuming Y points up); if False, it's
                placed at their maximum Y.
            xlength: Extent of the checkerboard along X.
            ylength: Extent of the checkerboard along its (pre-rotation) Y
                axis (i.e. depth once placed in the XZ plane).
            square_size: Side length of each individual checker square.

        Returns:
            A new, initialized ``CheckerBoard`` instance.
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
        """Build a checkerboard mesh in the XY plane.

        Tiles the ``xlength`` x ``ylength`` area with ``square_size`` squares,
        alternating black/white based on tile parity, each square built from
        two triangles.

        Args:
            black: RGB color (as a normalized array) for "black" squares.
            white: RGB color (as a normalized array) for "white" squares.
            square_size: Side length of each square.
            xlength: Total extent tiled along X.
            ylength: Total extent tiled along Y.

        Returns:
            A ``psbody.mesh.Mesh`` with vertices, faces, and per-face colors
            (``fc``) set, re-centered so the tiled area is roughly centered
            on the origin (via a fixed ``[-5, -5, 0]`` translation).
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
        """Return the rotation matrix that maps the XY plane to *plane*.

        Args:
            plane: Target plane, currently only ``'xz'`` is supported.

        Returns:
            (3, 3) rotation matrix.

        Raises:
            NotImplementedError: If ``plane == 'yz'`` (not yet implemented).
                Any other unrecognized value silently falls through and
                returns the identity matrix, matching the original behavior.
        """
        rot = np.eye(3)
        if plane == 'xz':
            rot[1, 1] = rot[2, 2] = 0
            rot[1, 2] = -1
            rot[2, 1] = 1
        elif plane == 'yz':
            raise NotImplementedError("'yz' plane not yet implemented.")
        return rot

    def _prep_for_render(self, checker: Mesh):
        """Convert a psbody ``Mesh`` into batched GPU tensors for rendering.

        Args:
            checker: A ``psbody.mesh.Mesh`` with vertices (``v``), faces
                (``f``), and per-face colors (``fc``) populated.

        Returns:
            A tuple of:
                verts: (1, N, 3) float32 CUDA tensor of vertices.
                faces: (1, F, 3) int CUDA tensor of face indices.
                texts: (1, F, 4, 4, 4, 3) float32 CUDA tensor holding each
                    face's flat color, broadcast across a 4x4x4 texture grid
                    (the layout expected by the downstream batched renderer).
        """
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
        """Set up the pyrender scene, camera, and (offscreen or windowed) viewer.

        Args:
            width: Render/viewport width in pixels.
            height: Render/viewport height in pixels.
            intrinsic: Initial camera intrinsics, indexed like a 3x3 matrix:
                ``intrinsic[0, 0]`` = fx, ``intrinsic[1, 1]`` = fy,
                ``intrinsic[0, 2]`` = cx, ``intrinsic[1, 2]`` = cy.
            use_offscreen: If True, creates a ``pyrender.OffscreenRenderer``
                with Raymond lighting; if False, opens an interactive
                ``pyrender.Viewer`` window running in its own thread.
        """
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
        """Update the scene camera's intrinsics and world pose in place.

        Args:
            camera_pose: (4, 4) world-space pose matrix for the camera.
            intrinsic: 3x3-indexed intrinsics matrix, read the same way as
                in ``__init__`` (``[0,0]``=fx, ``[1,1]``=fy, ``[0,2]``=cx,
                ``[1,2]``=cy).
        """
        cam = self.camera_node.camera
        cam.fx = intrinsic[0, 0]
        cam.fy = intrinsic[1, 1]
        cam.cx = intrinsic[0, 2]
        cam.cy = intrinsic[1, 2]
        self.scene.set_pose(self.camera_node, pose=camera_pose)

    def get_camera(self):
        """Return the scene's active ``pyrender`` camera object.

        Returns:
            The ``pyrender`` camera instance attached to ``self.camera_node``.
        """
        return self.camera_node.camera

    def get_projection_matrix(self):
        """Compute the current camera's projection matrix.

        Returns:
            The projection matrix for the active camera at the viewer's
            configured ``width``/``height``.
        """
        return self.camera_node.camera.get_projection_matrix(
            width=self.width, height=self.height
        )

    def remove_mesh(self):
        """Remove every mesh node (any node whose name contains ``'mesh'``) from the scene."""
        for node in self.scene.get_nodes():
            if node.name is not None and 'mesh' in node.name:
                self.scene.remove_node(node)


def init_mesh_viewer(camera_path: str) -> MeshViewer2:
    """Instantiate a ``MeshViewer2`` sized/calibrated from a reference camera XML.

    Reads intrinsics and image dimensions from ``Camera_1.xml`` under
    ``camera_path`` and uses them to construct a matching offscreen viewer.

    Args:
        camera_path: Directory containing ``Camera_1.xml`` (and, typically,
            the other calibrated cameras used elsewhere in the pipeline).

    Returns:
        A new ``MeshViewer2`` configured with that camera's intrinsics and
        image size, using offscreen rendering.
    """
    from data.yolo_data_gen import extract_from_xml  # local import to avoid circular deps

    K, _, (width, height) = extract_from_xml(os.path.join(camera_path, 'Camera_1.xml'))
    return MeshViewer2(width=width, height=height, intrinsic=K, use_offscreen=True)