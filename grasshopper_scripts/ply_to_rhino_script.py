#! python 3
# venv: 3d-photo
# r: numpy
# r: plyfile
import sys

utils_folders = [
    "/Users/devonperoutky/Development/projects/3d-photo/utils/",
    "/Users/devonperoutky/Development/projects/3d-photo/utils/custom_types/",
]

# Add to the folder importable paths if it's not already in there
for utils_folder in utils_folders:
    if utils_folder not in sys.path:
        sys.path.append(utils_folder)

import System
import Rhino
import Grasshopper
import rhinoscriptsyntax as rs

import Rhino.Geometry as RG

import System.Drawing as SD
import plyfile
import random
import numpy as np
import math
from typing import List, Tuple, Optional
from numpy.typing import NDArray

# Import the classes and functions
from gaussian_splat import GaussianSplat
from test_data import create_example_splats


class GaussianSplatReader:
    def create_splat_object_in_rhino(self, splat: GaussianSplat) -> RG.Brep:
        """Create a Rhino Brep object from a GaussianSplat instance.
        This method creates an ellipsoid based on the splat parameters.
        It uses the position, scale, and rotation to define the ellipsoid's geometry.

        Args:
            splat (GaussianSplat): The GaussianSplat instance containing parameters.
        Returns:
            RG.Brep: The resulting Brep object representing the splat.
        """
        # Create a unit sphere at origin
        sphere = RG.Sphere(RG.Point3d.Origin, 1.0)

        # ----------------------------------------------
        # 1. Shape - Scale the sphere to create an ellipsoid
        # ----------------------------------------------
        # Convert sphere to NurbsSurface for proper non-uniform scaling
        nurbs_surface = sphere.ToNurbsSurface()

        # Apply scaling transformation to the NURBS surface, as Berp cannot scale anisotropically
        scale_xf = RG.Transform.Scale(
            RG.Plane.WorldXY, splat.scale.X, splat.scale.Y, splat.scale.Z
        )
        nurbs_surface.Transform(scale_xf)

        # ----------------------------------------------
        # 2. Position - Apply translation to move to splat position
        # ----------------------------------------------
        translation = RG.Transform.Translation(RG.Vector3d(splat.position))
        nurbs_surface.Transform(translation)

        # ----------------------------------------------
        # 3. TODO: Rotation - Apply rotation to orient the splat
        # ----------------------------------------------

        # ----------------------------------------------
        # 4. Convert spherical harmonics DC coefficients to RGB [0-255]
        # ----------------------------------------------

        # ----------------------------------------------
        # 5. Convert spherical harmonics DC coefficients to RGB [0-255]
        # ----------------------------------------------

        # Convert back to Brep
        brep = nurbs_surface.ToBrep()

        return brep

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

        splats = []
        for v in verts:
            # FIX 1: Transform scale values from log space to real space
            # Scale values are stored as logarithms (range: -7 to -0.25)
            # Real scales are obtained by applying exp() function (range: ~0.02 to 0.35)
            scale_real = RG.Vector3d(
                math.exp(float(v["scale_0"])),  # exp(-7) ≈ 0.0009, exp(-0.25) ≈ 0.78
                math.exp(float(v["scale_1"])),
                math.exp(float(v["scale_2"])),
            )

            # FIX 2: Normalize quaternion rotation values
            # Quaternions must have unit length (norm = 1.0) for valid rotations
            # Raw quaternions from PLY may not be normalized (69% have norm ≠ 1.0)
            quat_raw = np.array(
                (v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]), dtype=np.float32
            )
            quat_norm = np.linalg.norm(quat_raw)
            if quat_norm > 0:
                quat_normalized = quat_raw / quat_norm  # Normalize to unit length
            else:
                quat_normalized = quat_raw  # Handle edge case of zero quaternion

            # FIX 3: Transform opacity from logit space to probability space
            # Opacity is stored in logit space (unbounded real numbers)
            # Convert to [0,1] probability using sigmoid function: 1/(1+exp(-x))
            opacity_logit = float(v["opacity"])
            opacity_real = 1.0 / (
                1.0 + math.exp(-opacity_logit)
            )  # Sigmoid transformation

            splats.append(
                GaussianSplat(
                    position=RG.Point3d(float(v["x"]), float(v["y"]), float(v["z"])),
                    scale=scale_real,  # Now using properly transformed scales
                    rotation_angles=quat_normalized,  # Now using normalized quaternions
                    color=np.array(
                        (v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]), dtype=np.float32
                    ),
                    opacity=opacity_real,  # Now using properly transformed opacity
                    normal=np.array((v["nx"], v["ny"], v["nz"]), dtype=np.float32),
                )
            )

        return splats

    def run(
        self,
        file_path: str,
        scale_factor: float,
        subdivision_level: int,
        sample_percentage: float,
    ):
        print("=== RunScript STARTED ===")

        # Add your processing logic here
        print(
            f"Processed with inputs: {file_path}, {scale_factor}, {subdivision_level}, {sample_percentage}"
        )

        # 1. Read PLY file
        splat_data = self.load_gaussian_splats(file_path)

        print(f"Loaded {len(splat_data)} total gaussian splats")

        splat_data = random.sample(
            splat_data, round(len(splat_data) * sample_percentage)
        )
        print(f"Using {len(splat_data)} gaussian splats")
        points, colors = [], []

        # Normalize position around origin
        average_x_position = np.mean([splat.position.X for splat in splat_data])
        avg_y_position = np.mean([splat.position.Y for splat in splat_data])
        avg_z_position = np.mean([splat.position.Z for splat in splat_data])

        splat_data = [
            GaussianSplat(
                position=RG.Point3d(
                    splat.position.X - average_x_position,
                    splat.position.Y - avg_y_position,
                    splat.position.Z - avg_z_position,
                ),
                scale=splat.scale,
                rotation_angles=splat.rotation_angles,
                color=splat.color,
                opacity=splat.opacity,
                normal=splat.normal,
            )
            for splat in splat_data
        ]

        for splat in splat_data:
            # brep = self.create_splat_object_in_rhino(splat)
            # breps.append(brep)
            # For now, just create a point at the splat position
            # Apply coordinate system correction - rotate 90 degrees around X-axis and flip Z
            # Original: (X, Y, Z) -> Rotated: (X, Z, -Y)
            point = RG.Point3d(splat.position.X, splat.position.Z, -splat.position.Y)
            points.append(point)

            r = self.sh_to_rgb(splat.color[0])  # Red channel from f_dc_0
            g = self.sh_to_rgb(splat.color[1])  # Green channel from f_dc_1
            b = self.sh_to_rgb(splat.color[2])  # Blue channel from f_dc_2

            color = SD.Color.FromArgb(r, g, b)
            colors.append(color)

        # Return geometry and colors for Grasshopper to display
        print("=== RunScript COMPLETED ===")
        print(f"Total Points created: {len(points)}")
        return points, colors

    def sh_to_rgb(self, sh_coeff):
        """
        Convert spherical harmonics DC coefficients to RGB color values.
        This function applies a sigmoid transformation to the spherical harmonics
        from unbounded real values to a [0,1] range, then scales to [0,255]
        """
        # Apply sigmoid activation: 1 / (1 + exp(-sh_coeff))
        # This converts from spherical harmonics space to probability space [0,1]
        sigmoid = 1.0 / (1.0 + math.exp(-sh_coeff))
        # Scale from [0,1] to [0,255] range for RGB color values
        return int(sigmoid * 255)


workflow_manager = GaussianSplatReader()

# This variables should be set by the Grasshopper environment
results = workflow_manager.run(
    file_path=file_path,
    scale_factor=scale_factor,
    subdivision_level=subdivision_level,
    sample_percentage=sample_percentage,
)

breps = results[0]  # Geometry output
colors = results[1]  # Colors output
