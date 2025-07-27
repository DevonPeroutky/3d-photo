#! python 3
# venv: 3d-photo
# r: numpy
# r: plyfile

import System
import Rhino
import Grasshopper
import rhinoscriptsyntax as rs

import Rhino.Geometry as RG

import System.Drawing as SD
import plyfile
import numpy as np
import math
from typing import List, Tuple, Optional
from numpy.typing import NDArray


class GaussianSplat:
    """Simple data structure to hold Gaussian splat parameters"""

    def __init__(
        self,
        position: RG.Point3d,
        scale: RG.Vector3d,
        rotation_angles: NDArray[np.float32],
        color: NDArray[np.float32],
        opacity: float,
        normal: Optional[NDArray[np.float32]] = None,
    ):
        self.position = position  # Point3d - center position
        self.scale = scale  # Vector3d - radii in X,Y,Z directions
        self.rotation_angles = (
            rotation_angles  # tuple (w, x, y, z) quaternion components
        )
        self.color = color  # tuple (R,G,B) 0-255
        self.opacity = opacity  # float 0.0-1.0
        self.normal = normal

    def __str__(self):
        return (
            f"GaussianSplat(position={self.position}, scale={self.scale}, "
            f"rotation_angles={self.rotation_angles}, color={self.color}, "
            f"opacity={self.opacity})"
        )


class GaussianSplatReader:
    def quaternion_to_transform(self, quat: NDArray[np.float32]) -> RG.Transform:
        """Convert quaternion (w, x, y, z) to Rhino Transform"""
        w, x, y, z = quat

        # Create Rhino quaternion (expects w, x, y, z order)
        rhino_quat = RG.Quaternion(w, x, y, z)

        # Use the Transform version of GetRotation with out parameter
        transform = RG.Transform.Identity
        rhino_quat.GetRotation(transform)
        return transform

    def create_splat_ellipsoid(self, splat: GaussianSplat) -> RG.Brep:
        """Create a NURBS ellipsoid from splat parameters"""
        print(splat)

        # Create base ellipsoid at origin
        base_plane = RG.Plane.WorldXY
        # ellipsoid = RG.Ellipsoid(base_plane, splat.scale.X, splat.scale.Y, splat.scale.Z)
        # brep = ellipsoid.ToBrep()

        # Create a sphere at origin
        print(RG.Point3d.Origin)
        sphere = RG.Sphere(splat.position, 2.0)
        brep = RG.Brep.CreateFromSphere(sphere)

        # Apply scaling first
        # scale_xf = RG.Transform.Scale(
        #     base_plane, splat.scale.X, splat.scale.Y, splat.scale.Z
        # )

        # Create transformation matrix
        # transform = RG.Transform.Identity
        # transform = transform * scale_xf

        # Apply quaternion rotation
        # assert len(splat.rotation_angles) == 4, (
        #     "Rotation angles must be a quaternion (w, x, y, z)"
        # )
        # rotation_transform = self.quaternion_to_transform(splat.rotation_angles)
        # transform = transform * rotation_transform

        # Apply translation
        # translation = RG.Transform.Translation(RG.Vector3d(splat.position))
        # transform = transform * translation

        # Transform the geometry
        # brep.Transform(transform)

        return brep

    def create_example_splats(self) -> List[GaussianSplat]:
        """Create some example Gaussian splats for testing"""

        splats = []

        # Splat 1: Red, centered at origin
        splats.append(
            GaussianSplat(
                position=RG.Point3d(0, 0, 0),
                scale=RG.Vector3d(2, 1, 1),
                rotation_angles=(
                    math.cos(math.pi / 8),
                    0,
                    0,
                    math.sin(math.pi / 8),
                ),  # 45 degree rotation around Z as quaternion
                color=(255, 100, 100),
                opacity=0.7,
            )
        )

        # Splat 2: Green, offset position
        splats.append(
            GaussianSplat(
                position=RG.Point3d(5, 0, 2),
                scale=RG.Vector3d(1, 3, 0.5),
                rotation_angles=(
                    math.cos(math.pi / 12),
                    math.sin(math.pi / 12),
                    0,
                    0,
                ),  # 30 degree rotation around X as quaternion
                color=(100, 255, 100),
                opacity=0.8,
            )
        )

        # Splat 3: Blue, different orientation
        splats.append(
            GaussianSplat(
                position=RG.Point3d(-3, 4, -1),
                scale=RG.Vector3d(1.5, 1.5, 2),
                rotation_angles=(
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                ),  # Complex rotation as quaternion (normalized)
                color=(100, 100, 255),
                opacity=0.6,
            )
        )

        return splats

    def load_gaussian_splats(self, file_path: str) -> List[GaussianSplat]:
        """Read the PLY above and return a list of GaussianSplat objects.
        # 3D world coordinates
        property float x:
        property float y
        property float z

        # Normal unit vector components, indicating the surface orientation at each point
        property float nx
        property float ny
        property float nz

        # Spherical harmonics coefficients for the DC term representing the base RGB color
        property float f_dc_0
        property float f_dc_1
        property float f_dc_2

        # Transparency
        property float opacity

        # 3D Shape Parameters: Anisotropic scaling factors along the three principal axes
        property float scale_0
        property float scale_1
        property float scale_2

        # 3D Orientation: Quaternion rotation components
        property float rot_0
        property float rot_1
        property float rot_2
        property float rot_3
        """
        ply = plyfile.PlyData.read(file_path)
        verts = ply["vertex"].data  # structured numpy array

        return [
            GaussianSplat(
                position=RG.Point3d(float(v["x"]), float(v["y"]), float(v["z"])),
                scale=RG.Vector3d(
                    float(v["scale_0"]), float(v["scale_1"]), float(v["scale_2"])
                ),
                rotation_angles=np.array(
                    (v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]), dtype=np.float32
                ),
                color=np.array(
                    (v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]), dtype=np.float32
                ),
                opacity=float(v["opacity"]),
                normal=np.array((v["nx"], v["ny"], v["nz"]), dtype=np.float32),
            )
            for v in verts
        ]

    def run(
        self,
        file_path: str,
        scale_factor: float,
        subdivision_level: int,
        max_splats: int,
    ):
        print("=== RunScript STARTED ===")

        # Add your processing logic here
        print(
            f"Processed with inputs: {file_path}, {scale_factor}, {subdivision_level}, {max_splats}"
        )

        splats: List[GaussianSplat] = self.create_example_splats()
        breps = []
        colors = []

        print(f"Creating {len(splats)} example splats...")
        for splat in splats:
            # Create the ellipsoid geometry
            brep: RG.Brep = self.create_splat_ellipsoid(splat)
            breps.append(brep)

            # Store color information for Grasshopper to use
            # Convert RGB (0-255) to RGB (0-1) for Grasshopper
            color = SD.Color.FromArgb(splat.color[0], splat.color[1], splat.color[2])
            color = SD.Color.FromArgb(0, 0, 0)
            colors.append(color)

        print(f"Created {len(breps)} splats")

        # 1. Read PLY file
        # splat_data = self.load_gaussian_splats(file_path)

        # print(f"Loaded {len(splat_data)} gaussian splats")

        # for splat in splat_data:
        #     print(splat)
        #     brep = self.create_splat_ellipsoid(splat)
        #     breps.append(brep)
        #
        #     color = SD.Color.FromArgb(
        #         int(splat.color[0]), int(splat.color[1]), int(splat.color[2])
        #     )
        #     colors.append(color)

        # 2. Parse Gaussian splat parameters
        # positions, scales, rotations, colors = self._parse_splat_data(splat_data, max_splats)

        # 3. Generate SubD spheres
        # spheres = self._create_subd_spheres(positions, scales, scale_factor, subdivision_level)

        # 4. Prepare metadata
        # metadata = self._create_metadata(splat_data, len(spheres))

        # Return geometry and colors for Grasshopper to display
        print("=== RunScript COMPLETED ===")
        return breps, colors


workflow_manager = GaussianSplatReader()

# This variables should be set by the Grasshopper environment
results = workflow_manager.run(
    file_path=file_path,
    scale_factor=scale_factor,
    subdivision_level=subdivision_level,
    max_splats=max_splats,
)

breps = results[0]  # Geometry output
colors = results[1]  # Colors output
