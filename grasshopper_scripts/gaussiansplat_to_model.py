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
from typing import List, Tuple, Optional, Iterator
from numpy.typing import NDArray
from itertools import accumulate, chain
from functools import reduce

# Import local classes and functions
from performance_monitor import PerformanceMonitor
from custom_types import GaussianSplat
from color_utils import ColorUtils
from gaussian_splat_reader import GaussianSplatReader


def consolidate_meshes(sphere_meshes: List[Tuple[RG.Mesh, List[SD.Color]]]) -> RG.Mesh:
    """
    Consolidates a list of meshes and their corresponding vertex colors into a single
    merged mesh.

    Args:
        sphere_meshes: A list of tuples, where each tuple contains a Rhino.Geometry.Mesh
                       and a list of System.Drawing.Color objects for its vertices.

    Returns:
        A single, merged Rhino.Geometry.Mesh with all the geometry and vertex colors
        of the input meshes.
    """
    all_vertices = []
    all_faces = []
    all_vertex_colors = []
    vertex_offset = 0

    for mesh, colors in sphere_meshes:
        # Add vertices
        all_vertices.extend(mesh.Vertices)

        # Add vertex colors
        all_vertex_colors.extend(colors)

        for face in mesh.Faces:
            new_face = RG.MeshFace(
                face.A + vertex_offset,
                face.B + vertex_offset,
                face.C + vertex_offset,
                face.D + vertex_offset if face.IsQuad else face.C + vertex_offset,
            )
            all_faces.append(new_face)

        # Update the vertex offset for the next mesh
        vertex_offset += mesh.Vertices.Count

    # Create the final merged mesh
    merged_mesh = RG.Mesh()
    merged_mesh.Vertices.AddVertices(all_vertices)
    merged_mesh.Faces.AddFaces(all_faces)
    merged_mesh.VertexColors.SetColors(all_vertex_colors)

    # Perform final cleanup
    merged_mesh.Normals.ComputeNormals()
    merged_mesh.Compact()
    merged_mesh.UnifyNormals()
    merged_mesh.RebuildNormals()

    return merged_mesh


def create_single_mesh(
    splat: GaussianSplat, sphere_template: RG.Mesh
) -> Tuple[RG.Mesh, List[SD.Color]]:
    sphere = sphere_template.Duplicate()

    # Scale the sphere based on the splat's scale
    scale_factor = splat.scale

    # print(splat.scale.X)
    scale_transform = RG.Transform.Scale(
        RG.Plane.WorldXY,
        float(splat.scale.X),
        float(splat.scale.Y),
        float(splat.scale.Z),
    )
    sphere.Transform(scale_transform)

    transform = RG.Transform.Translation(
        RG.Vector3d(
            splat.position.X,
            splat.position.Z,
            -splat.position.Y,
        )
    )
    sphere.Transform(transform)

    # Create color for this splat
    r = ColorUtils.sh_to_rgb(splat.color[0])
    g = ColorUtils.sh_to_rgb(splat.color[1])
    b = ColorUtils.sh_to_rgb(splat.color[2])
    alpha = int(splat.opacity * 255)
    color = SD.Color.FromArgb(alpha, r, g, b)

    # Add vertex colors for all vertices of this sphere
    colors = [color for i in range(sphere.Vertices.Count)]

    return sphere, colors


def create_merged_mesh(splats: List[GaussianSplat]) -> RG.Mesh:
    # Cache a unit sphere at origin
    unit_sphere = RG.Mesh.CreateFromSphere(RG.Sphere(RG.Point3d.Origin, 1), 4, 4)

    sphere_meshes: List[Tuple[RG.Mesh, List[SD.Color]]] = [
        create_single_mesh(splat, unit_sphere) for splat in splats
    ]

    # Consolidate all meshes and their vertex colors into a single mesh
    return consolidate_meshes(sphere_meshes)


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
    # Calculate centroid
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
subdivision_level = int(
    globals().get("subdivision_level", 3)
)  # Default subdivision level
sample_percentage = globals().get("sample_percentage", 1.0)  # Default to 100%
render_mode = globals().get("render_mode", "preview")  # Default render mode


# Read the Gaussian splat data from the PLY file
splat_reader = GaussianSplatReader()
splat_data = splat_reader.load_gaussian_splats(file_path)
centroid_point, centroid_cube, centroid_color = visualize_centroid(splat_data)
splat_data = splat_reader.sample_by_region(splat_data, centroid_point, 2)
print(f"Loaded {len(splat_data)} Gaussian splats from {file_path}")

# Center splats around the origin
splat_data = splat_reader.normalize_splat_position_to_origin(splat_data)

# Render unmodified pointcloud to visualize the original splats
og_pointcloud = render_point_cloud(
    splat_data,
    position_offset=RG.Vector3d(0, 0, 4),  # No offset for original point cloud
)

# Converstion pipeline
# 1. Filter splats
filter_config = {
    "distance_centroid": {"enabled": True, "percentile": 99.0},
    "opacity": {
        "enabled": True,
        "min_opacity": 0.1,
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

merged_mesh = create_merged_mesh(splat_data)

geometries = [og_pointcloud, merged_mesh]
colors = [SD.Color.White, SD.Color.Black]
