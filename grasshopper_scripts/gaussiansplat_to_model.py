#! python 3
# venv: 3d-photo
# r: numpy
# r: plyfile
# r: psutil
import sys

# Add to the folder importable paths if it's not already in there
utils_folders = [
    "/Users/devonperoutky/Development/projects/3d-photo/utils/",
    "/Users/devonperoutky/Development/projects/3d-photo/custom_types/",
]
for utils_folder in utils_folders:
    if utils_folder not in sys.path:
        sys.path.append(utils_folder)

# Reload custom modules for development (ensures changes are picked up)
import importlib

try:
    # Import the reloader utility
    from module_reloader import reload_all_custom_modules

    # Reload all custom modules - set debug=True to see reload status
    reload_all_custom_modules(debug=False)
except Exception as e:
    print(f"Module reload failed: {e}")

import System
import Rhino
import Grasshopper
import rhinoscriptsyntax as rs
import Rhino.Geometry as RG
import System.Drawing as SD
from System import Array, Double, Int32, Single
import plyfile
import random
import numpy as np
import math
from typing import List, Tuple, Optional, Iterator
from numpy.typing import NDArray
from itertools import accumulate, chain
from functools import reduce

# Import local classes and functions
from performance_monitor import PerformanceMonitor
from custom_types import GaussianSplat, QuatFormat
from color_utils import ColorUtils
from gaussian_splat_reader import GaussianSplatReader


# TODO: Remove this function with Rhino 8 SDK.
def quaternion_to_rotation_transform_custom(
    quat_wxyz: NDArray[np.float64],
) -> RG.Transform:
    """
    Convert a quaternion (w, x, y, z) to a Rhino Transform (rotation about origin). The Mathematical Transformation

    # Quaternion to Rotation Matrix Conversion

        The function converts a quaternion to a 3x3 rotation matrix using this standard formula:

        R = [1-2(y²+z²)   2(xy-zw)   2(xz+yw)]
            [2(xy+zw)   1-2(x²+z²)   2(yz-xw)]
            [2(xz-yw)   2(yz+xw)   1-2(x²+y²)]

    ## How Each Matrix Element Works

        Diagonal Elements (R₀₀, R₁₁, R₂₂):
        - R.M00 = 1.0 - 2.0 * (y*y + z*z) - How much the X-axis stays along X
        - R.M11 = 1.0 - 2.0 * (x*x + z*z) - How much the Y-axis stays along Y
        - R.M22 = 1.0 - 2.0 * (x*x + y*y) - How much the Z-axis stays along Z

        Off-diagonal Elements:
        - Control how axes get mixed/rotated into each other
        - R.M01 = 2.0 * (x*y - z*w) - How much X-axis contributes to Y direction

    # 3D Rotation Mechanics

    ## How the Transform Rotates Geometry

        When you apply this transform to a 3D point P = (px, py, pz), it performs:

        P_rotated = R × P

        Step-by-step rotation process:

        1. X-component calculation:
        new_x = R.M00*px + R.M01*py + R.M02*pz
                = [1-2(y²+z²)]*px + [2(xy-zw)]*py + [2(xz+yw)]*pz
        2. Y-component calculation:
        new_y = R.M10*px + R.M11*py + R.M12*pz
                = [2(xy+zw)]*px + [1-2(x²+z²)]*py + [2(yz-xw)]*pz
        3. Z-component calculation:
        new_z = R.M20*px + R.M21*py + R.M22*pz
                = [2(xz-yw)]*px + [2(yz+xw)]*py + [1-2(x²+y²)]*pz

    ## Physical Interpretation

        The rotation matrix describes how the coordinate axes transform:

        - Row 0 [R.M00, R.M01, R.M02]: Where the original X-axis points after rotation
        - Row 1 [R.M10, R.M11, R.M12]: Where the original Y-axis points after rotation
        - Row 2 [R.M20, R.M21, R.M22]: Where the original Z-axis points after rotation

    ## Quaternion Rotation Axis and Angle

    A unit quaternion (w, x, y, z) represents:
    - Rotation axis: (x, y, z) (when normalized)
    - Rotation angle: θ = 2 * arccos(w)

    The quaternion encodes rotation by half-angle formulas:
    - w = cos(θ/2)
    - (x, y, z) = sin(θ/2) * axis_unit_vector

    ## Example Walkthrough

    Let's say you have a quaternion representing 90° rotation around Z-axis:
    - θ = 90° = π/2
    - w = cos(π/4) = √2/2
    - x = 0, y = 0, z = sin(π/4) = √2/2

    The resulting matrix would be:
    R = [0  -1   0]
        [1   0   0]
        [0   0   1]

    Applying this to point (1, 0, 0):
    - new_x = 0*1 + (-1)*0 + 0*0 = 0
    - new_y = 1*1 + 0*0 + 0*0 = 1
    - new_z = 0*1 + 0*0 + 1*0 = 0

    Result: (1, 0, 0) → (0, 1, 0) - X-axis rotated to Y-axis, exactly what we expect for 90° Z-rotation!

    Parameters
    ----------
    quat_wxyz : np.ndarray
        Quaternion as (w, x, y, z). Does not need to be unit-length.

    Returns
    -------
    Rhino.Geometry.Transform
        4x4 transform whose 3x3 block is the rotation matrix.
    """
    # Convert to numpy array and validate
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    assert quat_wxyz.size == 4, "quat_wxyz must be a length-4 array: (w, x, y, z)."

    # Quaternion component extraction block:
    # Converts numpy array elements to Python floats for mathematical operations
    # Quaternion format is (w, x, y, z) where w is the scalar (real) part
    # and (x, y, z) form the vector (imaginary) part
    w, x, y, z = map(float, quat_wxyz)

    # Quaternion normalization block:
    # Computes the magnitude (length) of the quaternion vector in 4D space
    # Formula: |q| = sqrt(w² + x² + y² + z²)
    # This is essential because only unit quaternions represent valid rotations
    n = math.sqrt(w * w + x * x + y * y + z * z)

    # Normalization safety check block:
    # Guards against degenerate quaternions that would cause division by zero
    # or invalid rotations. Zero-length quaternions don't represent any rotation
    if n == 0.0 or not math.isfinite(n):
        raise ValueError("Quaternion has zero or invalid length.")

    # Unit quaternion creation block:
    # Divides each component by the magnitude to create a unit quaternion
    # Unit quaternions have magnitude 1 and are the only valid rotation representations
    w, x, y, z = w / n, x / n, y / n, z / n

    # Transform matrix initialization block:
    # Creates a 4x4 identity matrix as the base for our rotation transform
    # Identity matrix ensures translation components remain zero (pure rotation)
    R = RG.Transform.Identity

    # First row of rotation matrix (R[0,:]) - X-axis transformation:
    # These values determine how the X-axis gets rotated in 3D space
    # Formula derived from quaternion-to-matrix conversion mathematics
    R.M00 = 1.0 - 2.0 * (y * y + z * z)  # X-axis X-component: 1 - 2(y² + z²)
    R.M01 = 2.0 * (x * y - z * w)  # X-axis Y-component: 2(xy - zw)
    R.M02 = 2.0 * (x * z + y * w)  # X-axis Z-component: 2(xz + yw)
    R.M03 = 0.0  # X-axis translation: 0 (pure rotation)

    # Second row of rotation matrix (R[1,:]) - Y-axis transformation:
    # These values determine how the Y-axis gets rotated in 3D space
    R.M10 = 2.0 * (x * y + z * w)  # Y-axis X-component: 2(xy + zw)
    R.M11 = 1.0 - 2.0 * (x * x + z * z)  # Y-axis Y-component: 1 - 2(x² + z²)
    R.M12 = 2.0 * (y * z - x * w)  # Y-axis Z-component: 2(yz - xw)
    R.M13 = 0.0  # Y-axis translation: 0 (pure rotation)

    # Third row of rotation matrix (R[2,:]) - Z-axis transformation:
    # These values determine how the Z-axis gets rotated in 3D space
    R.M20 = 2.0 * (x * z - y * w)  # Z-axis X-component: 2(xz - yw)
    R.M21 = 2.0 * (y * z + x * w)  # Z-axis Y-component: 2(yz + xw)
    R.M22 = 1.0 - 2.0 * (x * x + y * y)  # Z-axis Z-component: 1 - 2(x² + y²)
    R.M23 = 0.0  # Z-axis translation: 0 (pure rotation)

    # Fourth row of homogeneous matrix (R[3,:]) - Homogeneous coordinates:
    # This row maintains the homogeneous coordinate system properties
    # Required for proper 4x4 transformation matrix format in computer graphics
    R.M30 = 0.0  # No perspective transformation in X
    R.M31 = 0.0  # No perspective transformation in Y
    R.M32 = 0.0  # No perspective transformation in Z
    R.M33 = (
        1.0  # Homogeneous coordinate scaling factor (always 1 for affine transforms)
    )

    return R


def transform_quaternion_coordinate_system(
    quat_wxyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Transform quaternion from PLY coordinate system to Rhino coordinate system.

    Applies the same coordinate mapping used for positions: (X,Y,Z) → (X,Z,-Y)
    This ensures rotations are consistent with transformed positions.

    PLY coordinates: Y-up, X-right, Z-forward
    Rhino coordinates: Z-up, X-right, Y-forward

    Parameters
    ----------
    quat_wxyz : NDArray[np.float64]
        Input quaternion in WXYZ format [w, x, y, z] in PLY coordinate system

    Returns
    -------
    NDArray[np.float64]
        Transformed quaternion in WXYZ format [w, x, y, z] in Rhino coordinate system
    """
    # Convert to numpy array for mathematical operations
    q = np.asarray(quat_wxyz, dtype=np.float64)
    assert q.size == 4, "Quaternion must be 4-element array (w, x, y, z)"

    w, x, y, z = q

    # Apply coordinate system transformation: (X,Y,Z) → (X,Z,-Y)
    # For quaternions, this means:
    # - X component stays the same (rotation around X-axis)
    # - Y component becomes Z component (PLY Y-up becomes Rhino Z-up)
    # - Z component becomes -Y component (PLY Z-forward becomes Rhino -Y)
    # - W component (scalar) stays the same
    transformed_quat = np.array([w, x, z, -y], dtype=np.float64)

    return transformed_quat


def quaternion_to_rotation_transform(quat_wxyz: NDArray[np.float64]) -> RG.Transform:
    """
    Convert (w, x, y, z) → Rhino.Transform.
    Falls back to Identity if the quaternion is invalid.
    """
    w, x, y, z = map(float, quat_wxyz)
    q = RG.Quaternion(w, x, y, z)

    if not q.Unitize():  # returns False if the quat had zero length
        return RG.Transform.Identity

    rot = RG.Transform()
    success = q.GetRotation(rot)  # Pass the transform directly

    if not success:
        return RG.Transform.Identity

    return rot


def quaternion_to_rotation_transform_rhino(quat_wxyz):
    q = np.asarray(quat_wxyz, float)
    if q.size != 4:
        return RG.Transform.Identity

    # If your PLY is xyzw, reorder to wxyz
    if abs(q[0]) <= 0.5 and abs(q[3]) >= 0.5:
        q = q[[3, 0, 1, 2]]

    w, x, y, z = q.tolist()
    quat = RG.Quaternion(w, x, y, z)
    if not quat.Unitize():
        return RG.Transform.Identity

    origin = RG.Point3d.Origin
    try:
        # This is the correct overload
        return RG.Transform.Rotation(quat, origin)
    except Exception:
        # Fallback: axis–angle
        ok, angle, axis = quat.GetRotation()
        return (
            RG.Transform.Rotation(angle, axis, origin) if ok else RG.Transform.Identity
        )


def create_merged_mesh(
    splats: List[GaussianSplat],
    position_offset: Optional[RG.Vector3d] = RG.Vector3d(0, 0, 0),
    scale_factor: float = 1.0,
    sphere_resolution: int = 8,
) -> RG.Mesh:
    """
    Optimized version that avoids mesh duplication for each splat.
    Transforms vertices directly for much better performance.
    """
    # Create base sphere only once
    base_sphere = RG.Mesh.CreateFromSphere(
        RG.Sphere(RG.Point3d.Origin, 1), sphere_resolution, sphere_resolution
    )

    # Extract base sphere data once
    base_vertices = [v for v in base_sphere.Vertices]
    base_vertex_count = len(base_vertices)
    base_faces = [f for f in base_sphere.Faces]

    # Initialize final mesh
    final_mesh = RG.Mesh()
    all_colors = []
    vertex_offset = 0

    # Process each splat without duplication
    for splat in splats:
        # Step 1: Apply the splat's UNIQUE non-uniform scale.
        # Your loading function already correctly did math.exp(splat.scale).
        sx = splat.scale.X * scale_factor
        sy = splat.scale.Y * scale_factor
        sz = splat.scale.Z * scale_factor
        scale_transform = RG.Transform.Scale(RG.Plane.WorldXY, sx, sy, sz)

        # Step 2: Apply the splat's UNIQUE individual rotation.
        # Use the original, untransformed quaternion data.
        rotation_transform = quaternion_to_rotation_transform_custom(
            # transform_quaternion_coordinate_system(splat.rotation_angles)
            splat.rotation_angles
        )

        # Step 3: Combine scale and rotation. This orients the ellipsoid correctly.
        combined_transform = rotation_transform * scale_transform

        # BAD Translation
        # translation = RG.Transform.Translation(
        #     RG.Vector3d(splat.position.X, splat.position.Z, -splat.position.Y)
        # )

        # Step 4: Translate the ellipsoid to its final position in the original coordinate system.
        translation = RG.Transform.Translation(
            RG.Vector3d(splat.position.X, splat.position.Y, splat.position.Z)
        )

        # Final transform
        final_transform = translation * combined_transform

        # Transform and add vertices directly
        for vertex in base_vertices:
            pt = RG.Point3d(vertex.X, vertex.Y, vertex.Z)
            pt.Transform(final_transform)
            final_mesh.Vertices.Add(pt)

        # Add faces with correct vertex offsets
        for face in base_faces:
            new_face = RG.MeshFace(
                face.A + vertex_offset,
                face.B + vertex_offset,
                face.C + vertex_offset,
                face.D + vertex_offset if face.IsQuad else face.C + vertex_offset,
            )
            final_mesh.Faces.AddFace(new_face)

        # Color
        r = ColorUtils.sh_to_rgb(splat.color[0])
        g = ColorUtils.sh_to_rgb(splat.color[1])
        b = ColorUtils.sh_to_rgb(splat.color[2])
        alpha = int(splat.opacity * 255)
        color = SD.Color.FromArgb(alpha, r, g, b)

        # Add colors for this splat's vertices
        for _ in range(base_vertex_count):
            all_colors.append(color)

        vertex_offset += base_vertex_count

    # Step 5: Apply the ONE-TIME coordinate system rotation to the ENTIRE assembled mesh.
    # This correctly moves the model from a Y-up to Z-up (Rhino) system.
    y_up_to_z_up_transform = RG.Transform.Rotation(
        -math.pi / 2,  # -90 degrees (negative rotation to avoid flipping)
        RG.Vector3d.XAxis,
        RG.Point3d.Origin,
    )
    final_mesh.Transform(y_up_to_z_up_transform)

    # Set all colors at once
    final_mesh.VertexColors.SetColors(all_colors)

    # Apply position offset
    final_mesh.Transform(RG.Transform.Translation(position_offset))

    # Final cleanup
    final_mesh.Normals.ComputeNormals()
    final_mesh.Compact()

    return final_mesh


def render_point_cloud(
    splats: List[GaussianSplat],
    position_offset: Optional[RG.Vector3d] = RG.Vector3d(0, 0, 0),
) -> RG.PointCloud:
    """
    Renders a point cloud from a list of Gaussian splats.

    Args:
        splats (List[GaussianSplat]): List of Gaussian splats to render.
        position_offset (Optional[RG.Vector3d]): An optional offset to apply to the splat positions.

    Returns:
        RG.PointCloud: The rendered point cloud.
    """
    point_cloud = RG.PointCloud()
    for splat in splats:
        # Apply the position offset numpy vector if provided
        # x = splat.position.X + position_offset
        # y = splat.position.Y + position_offset
        # z = splat.position.Z + position_offset

        point = (
            RG.Point3d(splat.position.X, splat.position.Z, -splat.position.Y)
            + position_offset
        )

        alpha = int(splat.opacity * 255)
        red = ColorUtils.sh_to_rgb(splat.color[0])
        green = ColorUtils.sh_to_rgb(splat.color[1])
        blue = ColorUtils.sh_to_rgb(splat.color[2])
        color = SD.Color.FromArgb(
            alpha,
            red,
            green,
            blue,
        )

        point_cloud.Add(point, color)

    print(f"Point cloud created with {point_cloud.Count} points")
    return point_cloud


def visualize_centroid(splat_data: List[GaussianSplat]) -> RG.Brep:
    splat_data_centroid = np.mean(
        [
            [splat.position.X, splat.position.Y, splat.position.Z]
            for splat in splat_data
        ],
        axis=0,
    )
    centroid_point = RG.Point3d(
        splat_data_centroid[0],
        splat_data_centroid[1],
        splat_data_centroid[2],
    )

    # Create a 3D cube to visualize the centroid
    centroid_cube = RG.Sphere(centroid_point, 0.25).ToBrep()
    return (
        centroid_point,
        centroid_cube,
        SD.Color.Red,
    )


# This variables should be set by the Grasshopper environment
debug_mode = globals().get("debug_mode", True)  # Default to False if not provided
file_path = globals().get("file_path", "../assets/JAPAN.ply")
scale_factor = globals().get("scale_factor", 1)  # Default scale factor
subdivision_level = int(globals().get("subdivision_level", 3))
sample_percentage = globals().get("sample_percentage", 1.0)  # Default to 100%
render_mode = globals().get("render_mode", "preview")  # Default render mode

# Read the Gaussian splat data from the PLY file
splat_reader = GaussianSplatReader()
splat_data = splat_reader.load_gaussian_splats(file_path, quat_format=QuatFormat.WXYZ)
centroid_point, centroid_cube, centroid_color = visualize_centroid(splat_data)
# splat_data = splat_reader.sample_by_region(splat_data, centroid_point, 2)
print(f"Loaded {len(splat_data)} Gaussian splats from {file_path}")

# Center splats around the origin
splat_data = splat_reader.normalize_splat_position_to_origin(splat_data)

# Render unmodified pointcloud to visualize the original splats
og_pointcloud = render_point_cloud(
    splat_data,
    position_offset=RG.Vector3d(0, 0, 8),  # No offset for original point cloud
)

# Converstion pipeline
# 1. Filter splats
filter_config = {
    "distance_centroid": {"enabled": True, "percentile": 99.0},
    "opacity": {
        "enabled": True,
        "min_opacity": 0.5,
    },  # Filter out low opacity splats
    "brightness": {
        "enabled": True,
        "min_brightness": 0.00,
        "max_brightness": 0.81,
    },
    "scale": {
        "enabled": True,
        "min_scale_percentile": 2.0,  # More aggressive filtering
        "max_scale_percentile": 98.0,
    },
}
splat_data = splat_reader.apply_filters(splat_data, filter_config)
print(f"{splat_data[0].quat_format}: {splat_data[0].rotation_angles}")

geometries = [
    og_pointcloud,
    create_merged_mesh(splat_data, scale_factor=2, sphere_resolution=6),
]
colors = [SD.Color.White, SD.Color.Black]
