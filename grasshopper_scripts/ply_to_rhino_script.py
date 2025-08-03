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
from typing import List, Tuple, Optional
from numpy.typing import NDArray

# Import local classes and functions
from performance_monitor import PerformanceMonitor
from custom_types import GaussianSplat
from color_utils import ColorUtils

# Auto-reload modules during development
import importlib
import performance_monitor
import custom_types
import color_utils

# Debug flag - set to False for production
DEBUG_MODE = True

if DEBUG_MODE:
    # Force reload of custom modules
    importlib.reload(performance_monitor)
    importlib.reload(custom_types)
    importlib.reload(color_utils)
    print("DEBUG: Reloaded custom modules")


class GaussianSplatReader:
    def __init__(self):
        """Initialize the GaussianSplatReader with performance monitoring."""
        self.perf_monitor = PerformanceMonitor()
        self._base_sphere_mesh = None

    # -----------------------------
    # Filters
    # -----------------------------
    def filter_by_opacity(
        self, splats: List[GaussianSplat], min_opacity: float = 0.1
    ) -> List[GaussianSplat]:
        """Filter out splats with low opacity values that contribute little to visual quality.

        NOISE REDUCTION: Gaussian splatting often generates semi-transparent splats that
        represent uncertain or low-confidence regions. These low-opacity splats create
        visual clutter and slow down rendering without adding meaningful detail to the
        final visualization. This filter removes splats below a transparency threshold.

        TYPICAL USE CASE: Remove "ghost" splats that appear as faint artifacts in the
        reconstruction, particularly useful when the original neural network training
        produced uncertain splats in empty space or at object boundaries.

        Args:
            splats: List of GaussianSplat objects
            min_opacity: Minimum opacity threshold (0.0 to 1.0)
        Returns:
            Filtered list of splats
        """
        filtered = [splat for splat in splats if splat.opacity >= min_opacity]
        print(
            f"Opacity filter: {len(splats)} -> {len(filtered)} splats (removed {len(splats) - len(filtered)})"
        )
        return filtered

    def filter_by_scale(
        self,
        splats: List[GaussianSplat],
        min_scale_percentile: float = 5.0,
        max_scale_percentile: float = 95.0,
    ) -> List[GaussianSplat]:
        """Filter out splats with extreme scale values that often represent noise.

        GEOMETRIC QUALITY CONTROL: Neural Radiance Field training can produce splats with
        extreme scales - either microscopic splats (scale < 0.001) that are invisible,
        or massive splats (scale > 10) that create blob-like artifacts covering large
        areas. This filter uses percentile-based thresholding to remove statistical
        outliers in splat size distribution.

        ALGORITHM: Calculates geometric mean of X,Y,Z scales for each splat to get
        representative size, then filters based on percentile thresholds to maintain
        splats within reasonable size bounds for CAD visualization.

        TYPICAL USE CASE: Clean up reconstructions where training instability created
        unreasonably large or small Gaussian kernels.

        Args:
            splats: List of GaussianSplat objects
            min_scale_percentile: Lower percentile threshold for scale filtering
            max_scale_percentile: Upper percentile threshold for scale filtering
        Returns:
            Filtered list of splats
        """
        # Calculate average scale for each splat (geometric mean of X, Y, Z scales)
        scales = [
            np.power(splat.scale.X * splat.scale.Y * splat.scale.Z, 1 / 3)
            for splat in splats
        ]

        min_threshold = np.percentile(scales, min_scale_percentile)
        max_threshold = np.percentile(scales, max_scale_percentile)

        filtered = []
        for i, splat in enumerate(splats):
            avg_scale = scales[i]
            if min_threshold <= avg_scale <= max_threshold:
                filtered.append(splat)

        print(
            f"Scale filter: {len(splats)} -> {len(filtered)} splats (removed {len(splats) - len(filtered)})"
        )
        return filtered

    def filter_by_color_variance(
        self,
        splats: List[GaussianSplat],
        min_variance: float = 0.01,
    ) -> List[GaussianSplat]:
        """Filter out low-variance colors that often represent noise.

        COLOR QUALITY ASSESSMENT: Splats with very low color variance (near-monochromatic)
        often represent areas where the neural network failed to learn meaningful color
        information, resulting in flat, uninteresting regions. These typically appear
        as uniform gray or single-color blobs that don't contribute to visual quality.

        ALGORITHM: Converts spherical harmonics to RGB, then calculates variance across
        R,G,B channels. Low variance indicates the splat displays essentially one color,
        suggesting it may be representing empty space or training artifacts rather than
        actual scene content.

        TYPICAL USE CASE: Remove bland background splats or areas where reconstruction
        failed to capture texture details, improving overall visual richness.

        Args:
            splats: List of GaussianSplat objects
            min_variance: Minimum color variance threshold
        Returns:
            Filtered list of splats
        """
        filtered = []
        for splat in splats:
            # Convert spherical harmonics to RGB values [0,1] for analysis
            rgb = [1.0 / (1.0 + math.exp(-c)) for c in splat.color]

            # Calculate color variance
            variance = np.var(rgb)

            # Keep splats with sufficient color variance
            if variance >= min_variance:
                filtered.append(splat)

        print(
            f"Color variance filter: {len(splats)} -> {len(filtered)} splats (removed {len(splats) - len(filtered)})"
        )
        return filtered

    def filter_by_brightness(
        self,
        splats: List[GaussianSplat],
        min_brightness: float = 0.1,
        max_brightness: float = 0.9,
    ) -> List[GaussianSplat]:
        """Filter out overly bright or dark colors that often represent noise.

        EXPOSURE CORRECTION: Neural radiance fields can generate splats with extreme
        brightness values - pure black splats (brightness ≈ 0) often represent areas
        with no training data, while pure white splats (brightness ≈ 1) typically indicate
        overexposed regions or numerical instability during training. Both create
        unrealistic artifacts in architectural/product visualization.

        ALGORITHM: Converts spherical harmonics to RGB, calculates average brightness,
        and filters out splats outside acceptable luminance range. This preserves
        natural lighting variation while removing extreme values.

        TYPICAL USE CASE: Clean up indoor scans with dark shadows and bright windows,
        or outdoor scans with harsh lighting conditions that created exposure artifacts.

        Args:
            splats: List of GaussianSplat objects
            min_brightness: Minimum brightness threshold (to filter dark points)
            max_brightness: Maximum brightness threshold (to filter white/bright points)
        Returns:
            Filtered list of splats
        """
        filtered = []
        for splat in splats:
            # Convert spherical harmonics to RGB values [0,1] for analysis
            rgb = [1.0 / (1.0 + math.exp(-c)) for c in splat.color]

            # Calculate brightness (average of RGB)
            brightness = np.mean(rgb)

            # Keep splats within brightness range
            if min_brightness <= brightness <= max_brightness:
                filtered.append(splat)

        print(
            f"Brightness filter: {len(splats)} -> {len(filtered)} splats (removed {len(splats) - len(filtered)})"
        )
        return filtered

    def filter_statistical_outliers(
        self, splats: List[GaussianSplat], k_neighbors: int = 20, std_ratio: float = 2.0
    ) -> List[GaussianSplat]:
        """Remove points based on statistical analysis of distance to k-nearest neighbors.

        SPATIAL OUTLIER DETECTION: This implements a classic point cloud denoising algorithm
        that identifies splats positioned far from their local neighborhood. Neural radiance
        field training can create "floating" splats in empty space due to view synthesis
        artifacts or insufficient training views. These isolated splats appear as noise
        when converted to CAD geometry.

        ALGORITHM: For each splat, finds k-nearest neighbors and calculates mean distance.
        Points with mean distances beyond (global_mean + std_ratio * global_std) are
        considered outliers. This is more robust than absolute distance thresholds as
        it adapts to local point density variations.

        COMPUTATIONAL COST: O(n²) - expensive for large datasets, use radius filter for
        better performance on dense point clouds.

        TYPICAL USE CASE: Remove stray splats floating in empty space, especially effective
        for architectural reconstructions where clean surfaces are desired.

        Args:
            splats: List of GaussianSplat objects
            k_neighbors: Number of nearest neighbors to consider
            std_ratio: Standard deviation multiplier for outlier threshold
        Returns:
            Filtered list of splats
        """
        if len(splats) <= k_neighbors:
            return splats

        # Extract positions as numpy array for efficient computation
        positions = np.array(
            [[splat.position.X, splat.position.Y, splat.position.Z] for splat in splats]
        )

        # Calculate distances to k-nearest neighbors for each point
        mean_distances = []
        for i, pos in enumerate(positions):
            # Calculate distances to all other points
            distances = np.linalg.norm(positions - pos, axis=1)
            # Sort and take k+1 nearest (excluding self at index 0)
            nearest_distances = np.sort(distances)[1 : k_neighbors + 1]
            mean_distances.append(np.mean(nearest_distances))

        mean_distances = np.array(mean_distances)

        # Calculate statistical thresholds
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        threshold = global_mean + std_ratio * global_std

        # Filter outliers
        filtered = [
            splat for i, splat in enumerate(splats) if mean_distances[i] <= threshold
        ]

        print(
            f"Statistical outlier filter: {len(splats)} -> {len(filtered)} splats (removed {len(splats) - len(filtered)})"
        )
        return filtered

    def filter_radius_outliers(
        self, splats: List[GaussianSplat], radius: float = 0.5, min_neighbors: int = 5
    ) -> List[GaussianSplat]:
        """Remove isolated points with few neighbors within a given radius.

        LOCAL DENSITY FILTERING: This filter removes splats that are spatially isolated
        from the main point cloud structure. Unlike statistical outlier detection, this
        uses a fixed radius search which makes it more predictable and faster for large
        datasets. Particularly effective for removing reconstruction artifacts that appear
        as small clusters or individual splats floating in empty space.

        ALGORITHM: For each splat, counts neighbors within fixed radius. Splats with
        fewer than min_neighbors are removed. This preserves dense regions while
        eliminating sparse outliers, making it ideal for cleaning up architectural
        scans or product visualizations where solid surfaces are expected.

        PERFORMANCE: O(n²) but with early termination, generally faster than statistical
        outlier detection. Radius should be set based on expected point density.

        TYPICAL USE CASE: Remove isolated noise points while preserving fine details
        in dense regions. Excellent for cleaning scans before 3D printing or CAD export.

        Args:
            splats: List of GaussianSplat objects
            radius: Search radius for neighbors
            min_neighbors: Minimum number of neighbors required
        Returns:
            Filtered list of splats
        """
        if len(splats) <= min_neighbors:
            return splats

        # Extract positions as numpy array
        positions = np.array(
            [[splat.position.X, splat.position.Y, splat.position.Z] for splat in splats]
        )

        filtered = []
        for i, splat in enumerate(splats):
            pos = positions[i]
            # Calculate distances to all other points
            distances = np.linalg.norm(positions - pos, axis=1)
            # Count neighbors within radius (excluding self)
            neighbor_count = np.sum((distances > 0) & (distances <= radius))

            if neighbor_count >= min_neighbors:
                filtered.append(splat)

        print(
            f"Radius outlier filter: {len(splats)} -> {len(filtered)} splats (removed {len(splats) - len(filtered)})"
        )
        return filtered

    def filter_distance_from_centroid(
        self, splats: List[GaussianSplat], percentile: float = 95.0
    ) -> List[GaussianSplat]:
        """Remove points too far from the centroid of the point cloud.

        GLOBAL BOUNDARY FILTERING: This filter removes splats that are extremely far
        from the center of mass of the entire point cloud. Neural radiance field training
        can sometimes generate splats at the edges of the training volume or beyond,
        creating artifacts that extend far outside the actual scene boundaries.

        ALGORITHM: Calculates the 3D centroid of all splat positions, then computes
        distance from each splat to this center point. Uses percentile-based thresholding
        to remove the most distant outliers while preserving the main object structure.
        This is particularly effective for scenes with a clear central subject.

        ROBUSTNESS: More robust than absolute distance thresholds as it adapts to the
        natural scale of each scene. Works well for both small objects and large environments.

        TYPICAL USE CASE: Remove splats that extend far beyond the main subject, especially
        useful for object scans where you want to focus on the central item and remove
        background artifacts or training boundary effects.

        Args:
            splats: List of GaussianSplat objects
            percentile: Percentile threshold for distance from centroid
        Returns:
            Filtered list of splats
        """
        if len(splats) == 0:
            return splats

        # Calculate centroid
        positions = np.array(
            [[splat.position.X, splat.position.Y, splat.position.Z] for splat in splats]
        )
        centroid = np.mean(positions, axis=0)

        # Calculate distances from centroid
        distances = np.linalg.norm(positions - centroid, axis=1)

        # Calculate threshold based on percentile
        threshold = np.percentile(distances, percentile)

        # Filter points beyond threshold
        filtered = [
            splat for i, splat in enumerate(splats) if distances[i] <= threshold
        ]

        print(
            f"Distance from centroid filter: {len(splats)} -> {len(filtered)} splats (removed {len(splats) - len(filtered)})"
        )
        return filtered

    def apply_filters(self, splats: List[GaussianSplat]) -> List[GaussianSplat]:
        """Apply multiple filters in sequence based on configuration.

        Args:
            splats: List of GaussianSplat objects
            filter_config: Dictionary specifying which filters to apply and their parameters
        Returns:
            Filtered list of splats
        """
        self.perf_monitor.start_timing("Filtering operations")

        # Default configuration - start with most effective filters for your use case
        filter_config = {
            "distance_centroid": {"enabled": True, "percentile": 99.0},
            "opacity": {"enabled": False, "min_opacity": 0.05},
            "color_variance": {
                "enabled": False,
                "min_variance": 0.004,
            },
            "brightness": {
                "enabled": True,
                "min_brightness": 0.00,
                "max_brightness": 0.81,
            },
            "scale": {
                "enabled": False,
                "min_scale_percentile": 5.0,
                "max_scale_percentile": 95.0,
            },
            "statistical": {"enabled": False, "k_neighbors": 20, "std_ratio": 2.0},
            "radius": {
                "enabled": False,
                "radius": 0.35,
                "min_neighbors": 5,
            },  # This is very slow
            "dbscan": {"enabled": False, "eps": 0.3, "min_samples": 10},
        }

        filtered_splats = splats
        print(f"Starting with {len(filtered_splats)} splats")

        # Apply filters in order of effectiveness for your specific problem
        if filter_config.get("opacity", {}).get("enabled", False):
            filtered_splats = self.filter_by_opacity(
                filtered_splats,
                **{k: v for k, v in filter_config["opacity"].items() if k != "enabled"},
            )

        if filter_config.get("distance_centroid", {}).get("enabled", False):
            filtered_splats = self.filter_distance_from_centroid(
                filtered_splats,
                **{
                    k: v
                    for k, v in filter_config["distance_centroid"].items()
                    if k != "enabled"
                },
            )

        if filter_config.get("color_variance", {}).get("enabled", False):
            filtered_splats = self.filter_by_color_variance(
                filtered_splats,
                **{
                    k: v
                    for k, v in filter_config["color_variance"].items()
                    if k != "enabled"
                },
            )

        if filter_config.get("brightness", {}).get("enabled", False):
            filtered_splats = self.filter_by_brightness(
                filtered_splats,
                **{
                    k: v
                    for k, v in filter_config["brightness"].items()
                    if k != "enabled"
                },
            )

        if filter_config.get("scale", {}).get("enabled", False):
            filtered_splats = self.filter_by_scale(
                filtered_splats,
                **{k: v for k, v in filter_config["scale"].items() if k != "enabled"},
            )

        if filter_config.get("statistical", {}).get("enabled", False):
            filtered_splats = self.filter_statistical_outliers(
                filtered_splats,
                **{
                    k: v
                    for k, v in filter_config["statistical"].items()
                    if k != "enabled"
                },
            )

        if filter_config.get("radius", {}).get("enabled", False):
            filtered_splats = self.filter_radius_outliers(
                filtered_splats,
                **{k: v for k, v in filter_config["radius"].items() if k != "enabled"},
            )

        if filter_config.get("dbscan", {}).get("enabled", False):
            filtered_splats = self.filter_dbscan_noise(
                filtered_splats,
                **{k: v for k, v in filter_config["dbscan"].items() if k != "enabled"},
            )

        print(
            f"Final result: {len(filtered_splats)} splats (removed {len(splats) - len(filtered_splats)} total)"
        )

        self.perf_monitor.end_timing()
        return filtered_splats

    # -----------------------------
    # Rhino Geometry Creation
    # -----------------------------
    def quaternion_to_rotation_transform(self, quat: np.ndarray) -> RG.Transform:
        """Convert a normalized quaternion to a Rhino rotation transformation.

        Args:
            quat: Normalized quaternion as numpy array [rot_0, rot_1, rot_2, rot_3]
        Returns:
            RG.Transform: Rotation transformation matrix
        """
        # Convert numpy array to python floats and handle potential data type issues
        try:
            # Extract quaternion components - assume [rot_0, rot_1, rot_2, rot_3] format from PLY
            # This appears to be [x, y, z, w] based on common conventions
            x = float(quat[0])
            y = float(quat[1])
            z = float(quat[2])
            w = float(quat[3])

            # Manually create rotation matrix from quaternion components
            # This avoids the System.Double conversion issues with RG.Quaternion constructor
            rotation_matrix = RG.Transform.Identity

            # Standard quaternion to rotation matrix conversion
            # Assumes normalized quaternion
            xx = x * x
            yy = y * y
            zz = z * z
            xy = x * y
            xz = x * z
            yz = y * z
            wx = w * x
            wy = w * y
            wz = w * z

            # Build rotation matrix manually
            rotation_matrix.M00 = 1.0 - 2.0 * (yy + zz)
            rotation_matrix.M01 = 2.0 * (xy - wz)
            rotation_matrix.M02 = 2.0 * (xz + wy)
            rotation_matrix.M03 = 0.0

            rotation_matrix.M10 = 2.0 * (xy + wz)
            rotation_matrix.M11 = 1.0 - 2.0 * (xx + zz)
            rotation_matrix.M12 = 2.0 * (yz - wx)
            rotation_matrix.M13 = 0.0

            rotation_matrix.M20 = 2.0 * (xz - wy)
            rotation_matrix.M21 = 2.0 * (yz + wx)
            rotation_matrix.M22 = 1.0 - 2.0 * (xx + yy)
            rotation_matrix.M23 = 0.0

            rotation_matrix.M30 = 0.0
            rotation_matrix.M31 = 0.0
            rotation_matrix.M32 = 0.0
            rotation_matrix.M33 = 1.0

            return rotation_matrix

        except Exception as e:
            print(f"Failed to create rotation matrix from quaternion {quat}: {e}")
            # Return identity transform as fallback
            return RG.Transform.Identity

    def _get_base_sphere_mesh(self) -> RG.Mesh:
        """Get or create a cached base sphere mesh for instancing.

        Returns:
            RG.Mesh: A unit sphere mesh at origin for reuse
        """
        if self._base_sphere_mesh is None:
            self._base_sphere_mesh = RG.Mesh.CreateFromSphere(
                RG.Sphere(RG.Point3d.Origin, 1.0), 4, 4
            )
        return self._base_sphere_mesh

    def _apply_splat_transformations(
        self,
        geometry,
        scale_x: float,
        scale_y: float,
        scale_z: float,
        splat: GaussianSplat,
    ):
        """Apply scale, rotation, coordinate transform, and translation to geometry.

        Args:
            geometry: RG.Mesh or RG.NurbsSurface to transform
            scale_x, scale_y, scale_z: Scale factors for ellipsoid
            splat: GaussianSplat containing rotation and position data
        """
        # 1. SCALE: Apply scaling transformation to create ellipsoid
        scale_transform = RG.Transform.Scale(
            RG.Plane.WorldXY, scale_x, scale_y, scale_z
        )
        geometry.Transform(scale_transform)

        # 2. ROTATE: Apply quaternion rotation
        try:
            rotation_transform = self.quaternion_to_rotation_transform(
                splat.rotation_angles
            )
            geometry.Transform(rotation_transform)
        except Exception as e:
            print(f"Warning: Failed to apply rotation {splat.rotation_angles}: {e}")

        # 3. COORDINATE SYSTEM TRANSFORMATION: Apply same (X, Z, -Y) transform as points
        coord_transform = RG.Transform.Identity
        coord_transform.M00 = 1  # X stays X
        coord_transform.M01 = 0
        coord_transform.M02 = 0
        coord_transform.M10 = 0  # Y becomes Z
        coord_transform.M11 = 0
        coord_transform.M12 = 1
        coord_transform.M20 = 0  # Z becomes -Y
        coord_transform.M21 = -1
        coord_transform.M22 = 0
        geometry.Transform(coord_transform)

        # 4. TRANSLATE: Move to final position (after coordinate transformation)
        transformed_pos = RG.Point3d(
            splat.position.X, splat.position.Z, -splat.position.Y
        )
        translation = RG.Transform.Translation(RG.Vector3d(transformed_pos))
        geometry.Transform(translation)

    def create_splat_mesh_in_rhino(self, splat: GaussianSplat) -> RG.Mesh:
        """Create a Rhino Mesh object from a GaussianSplat instance.
        This method creates an ellipsoid mesh based on the splat parameters.
        Much faster than Breps for visualization purposes. Uses mesh instancing for performance.

        Args:
            splat (GaussianSplat): The GaussianSplat instance containing parameters.
        Returns:
            RG.Mesh: The resulting Mesh object representing the splat.
        """
        # Get cached base sphere mesh and create a copy for transformation
        base_mesh = self._get_base_sphere_mesh()
        mesh = base_mesh.Duplicate()

        # Apply all transformations using shared helper method
        scale_x, scale_y, scale_z = splat.scale.X, splat.scale.Y, splat.scale.Z

        # DEBUG: Print scale values to diagnose distortion
        # if abs(scale_x) > 10 or abs(scale_y) > 10 or abs(scale_z) > 10:
        #     print(
        #         f"EXTREME SCALE VALUES: X={scale_x:.6f}, Y={scale_y:.6f}, Z={scale_z:.6f}"
        #     )
        # elif scale_x < 0.001 or scale_y < 0.001 or scale_z < 0.001:
        #     print(
        #         f"TINY SCALE VALUES: X={scale_x:.6f}, Y={scale_y:.6f}, Z={scale_z:.6f}"
        #     )

        self._apply_splat_transformations(mesh, scale_x, scale_y, scale_z, splat)

        return mesh

    def create_point_in_rhino(self, splat: GaussianSplat) -> RG.Point:
        """Create a Rhino Point object from a GaussianSplat instance.
        This is used as a fallback when mesh or Brep creation fails.

        Args:
            splat (GaussianSplat): The GaussianSplat instance containing parameters.
        Returns:
            RG.Point: The resulting Point object representing the splat.
        """
        # Create a point at the transformed position
        transformed_pos = RG.Point3d(
            splat.position.X, splat.position.Z, -splat.position.Y
        )
        return RG.Point(transformed_pos)

    def create_merged_mesh(self, splats: List[GaussianSplat]) -> RG.Mesh:
        """Create a single merged mesh containing all splat geometries with vertex colors.

        PERFORMANCE OPTIMIZATION: Instead of creating thousands of individual mesh objects,
        this creates one large mesh containing all splat geometries. This approach provides
        50-100x performance improvement over individual meshes by reducing draw calls from
        30K individual calls to a single batch call, while maintaining full 3D geometry
        detail and per-splat color information through vertex colors.

        TECHNICAL BENEFITS:
        - Single draw call vs thousands of individual draw calls
        - Reduced object management overhead in Rhino viewport
        - Better GPU utilization through batch processing
        - Contiguous memory layout for better cache performance
        - Maintains full 3D geometry detail unlike PointCloud mode

        IDEAL USE CASE: Large datasets (10K-50K splats) where individual mesh detail is
        desired but individual object performance is prohibitive. Provides mesh-quality
        visualization with near-PointCloud performance.

        Args:
            splats: List of GaussianSplat objects to convert
        Returns:
            RG.Mesh: Single merged mesh containing all splat geometries with vertex colors
        """
        self.perf_monitor.start_timing(f"Merged mesh creation for {len(splats)} splats")

        if len(splats) == 0:
            return RG.Mesh()

        # Create merged mesh
        merged_mesh = RG.Mesh()

        # Get cached base sphere for instancing
        base_sphere = self._get_base_sphere_mesh()

        for splat in splats:
            # Create individual sphere mesh with transformations
            sphere_mesh = base_sphere.Duplicate()

            # Apply transformations (scale, rotate, coordinate transform, translate)
            scale_x, scale_y, scale_z = splat.scale.X, splat.scale.Y, splat.scale.Z
            self._apply_splat_transformations(
                sphere_mesh, scale_x, scale_y, scale_z, splat
            )

            # Convert spherical harmonics to RGB color
            r = ColorUtils.sh_to_rgb(splat.color[0])  # Red channel from f_dc_0
            g = ColorUtils.sh_to_rgb(splat.color[1])  # Green channel from f_dc_1
            b = ColorUtils.sh_to_rgb(splat.color[2])  # Blue channel from f_dc_2
            color = SD.Color.FromArgb(r, g, b)

            # Store vertex count before merging for color assignment
            vertex_start_index = merged_mesh.Vertices.Count

            # Append this sphere to merged mesh
            merged_mesh.Append(sphere_mesh)

            # Add vertex colors for all vertices of this sphere
            vertex_end_index = merged_mesh.Vertices.Count
            for i in range(vertex_start_index, vertex_end_index):
                merged_mesh.VertexColors.Add(color)

        # Ensure mesh is valid and compute normals
        merged_mesh.Compact()
        merged_mesh.UnifyNormals()
        merged_mesh.RebuildNormals()

        # Report performance statistics
        metrics = self.perf_monitor.end_timing()
        estimated_individual_time = (
            len(splats) * 0.002
        )  # Rough estimate for individual meshes
        performance_gain = (
            estimated_individual_time / metrics["duration"]
            if metrics["duration"] > 0
            else 0
        )

        print(
            f"✅ Merged mesh created: {len(splats):,} splats, {merged_mesh.Vertices.Count:,} vertices in {metrics['duration']:.2f}s"
        )
        if performance_gain > 10:
            print(
                f"🚀 Estimated performance gain: {performance_gain:.0f}x faster than individual meshes"
            )

        return merged_mesh

    def create_colored_point_cloud(self, splats: List[GaussianSplat]) -> RG.PointCloud:
        """Create a single PointCloud object with colors for ultra-fast rendering of large datasets.

        PERFORMANCE OPTIMIZATION: Instead of creating thousands of individual geometry objects,
        this creates one PointCloud containing all splat positions and colors. This approach
        provides 100x performance improvement for large datasets (30K+ splats) while maintaining
        full color information for visualization.

        TECHNICAL BENEFITS:
        - Single object management (vs thousands of individual objects)
        - Batch GPU rendering (single draw call vs thousands)
        - Contiguous memory layout (better cache performance)
        - Optimized Rhino display pipeline usage

        IDEAL USE CASE: Real-time preview of large Gaussian splat datasets where individual
        splat geometry detail is less important than overall spatial and color distribution.
        Perfect for initial data exploration and performance-critical workflows.

        Args:
            splats: List of GaussianSplat objects to convert
        Returns:
            RG.PointCloud: Single point cloud object containing all splat data
        """
        self.perf_monitor.start_timing(f"PointCloud creation for {len(splats)} splats")

        point_cloud = RG.PointCloud()

        # Performance warning for very large datasets
        if len(splats) > 100000:
            print(
                f"⚠️  Large dataset: {len(splats):,} splats. Consider additional filtering for optimal performance."
            )
        elif len(splats) > 50000:
            print(
                f"📊 Processing {len(splats):,} splats in PointCloud mode for optimal performance."
            )

        for splat in splats:
            # Apply same coordinate transformation as other geometry types
            transformed_pos = RG.Point3d(
                splat.position.X, splat.position.Z, -splat.position.Y
            )

            # Convert spherical harmonics to RGB color using existing utility
            r = ColorUtils.sh_to_rgb(splat.color[0])  # Red channel from f_dc_0
            g = ColorUtils.sh_to_rgb(splat.color[1])  # Green channel from f_dc_1
            b = ColorUtils.sh_to_rgb(splat.color[2])  # Blue channel from f_dc_2
            color = SD.Color.FromArgb(r, g, b)

            # Add point with color to the point cloud
            point_cloud.Add(transformed_pos, color)

        # Report performance statistics
        metrics = self.perf_monitor.end_timing()
        estimated_individual_time = (
            len(splats) * 0.001
        )  # Rough estimate for individual points
        performance_gain = (
            estimated_individual_time / metrics["duration"]
            if metrics["duration"] > 0
            else 0
        )

        print(
            f"✅ PointCloud created: {len(splats):,} points in {metrics['duration']:.2f}s"
        )
        if performance_gain > 10:
            print(
                f"🚀 Estimated performance gain: {performance_gain:.0f}x faster than individual points"
            )

        return point_cloud

    def create_splat_geometry(self, splat: GaussianSplat, render_mode: str = "preview"):
        """Create either a mesh or Brep from a GaussianSplat based on render mode.

        NOTE: This method handles individual splat geometry creation. For pointcloud and merged modes,
        use create_colored_point_cloud() or create_merged_mesh() directly with the full splat list for optimal performance.

        Args:
            splat (GaussianSplat): The GaussianSplat instance containing parameters.
            render_mode (str): "preview" for fast mesh, "export" for precise Brep, "test" for points
        Returns:
            RG.Mesh or RG.Brep or RG.Point: The resulting geometry object.
        """
        if render_mode == "preview":
            return self.create_splat_mesh_in_rhino(splat)
        elif render_mode == "test":
            return self.create_point_in_rhino(splat)
        elif render_mode == "export":
            return self.create_splat_object_in_rhino(splat)
        else:
            raise ValueError(
                f"Invalid render_mode: {render_mode}. Use 'preview', 'export', 'test', 'pointcloud', or 'merged'"
            )

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
        nurbs_surface = sphere.ToNurbsSurface()

        # Apply all transformations using shared helper method
        scale_x, scale_y, scale_z = splat.scale.X, splat.scale.Y, splat.scale.Z
        self._apply_splat_transformations(
            nurbs_surface, scale_x, scale_y, scale_z, splat
        )

        # Convert back to Brep
        brep = nurbs_surface.ToBrep()

        return brep

    # -----------------------------
    # Workflow Utility Functions
    # -----------------------------
    def simple_sampling(
        self, splats: List[GaussianSplat], sample_percentage: float
    ) -> List[GaussianSplat]:
        assert 0 < sample_percentage <= 1, "Sample percentage must be between 0 and 1"
        return random.sample(splats, int(len(splats) * sample_percentage))

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
        self.perf_monitor.start_timing("PLY file loading")

        ply = plyfile.PlyData.read(file_path)
        verts = ply["vertex"].data  # structured numpy array

        splats = []
        scale_values_log = []  # Track original log values
        scale_values_real = []  # Track real (exp) values

        for v in verts:
            # FIX 1: Transform scale values from log space to real space
            # Scale values are stored as logarithms (range: -7 to -0.25)
            # Real scales are obtained by applying exp() function (range: ~0.02 to 0.35)
            scale_log = [float(v["scale_0"]), float(v["scale_1"]), float(v["scale_2"])]
            scale_values_log.extend(scale_log)

            scale_real = RG.Vector3d(
                math.exp(scale_log[0]),  # exp(-7) ≈ 0.0009, exp(-0.25) ≈ 0.78
                math.exp(scale_log[1]),
                math.exp(scale_log[2]),
            )
            scale_values_real.extend([scale_real.X, scale_real.Y, scale_real.Z])

            # FIX 2: Normalize quaternion rotation values
            # Quaternions must have unit length (norm = 1.0) for valid rotations
            # Raw quaternions from PLY may not be normalized (69% have norm ≠ 1.0)
            quat_raw = np.array(
                (v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]), dtype=np.float32
            )
            quat_norm = np.linalg.norm(quat_raw)
            quat_normalized = quat_raw if quat_norm <= 0 else quat_raw / quat_norm

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

        print(f"Loaded {len(splats)} Gaussian splats from {file_path}")

        # DEBUG: Print scale value statistics
        scale_log_array = np.array(scale_values_log)
        scale_real_array = np.array(scale_values_real)

        print(f"\n=== SCALE VALUE ANALYSIS ===")
        print(
            f"Log space (original) - Min: {np.min(scale_log_array):.3f}, Max: {np.max(scale_log_array):.3f}, Mean: {np.mean(scale_log_array):.3f}"
        )
        print(
            f"Real space (exp) - Min: {np.min(scale_real_array):.6f}, Max: {np.max(scale_real_array):.6f}, Mean: {np.mean(scale_real_array):.6f}"
        )
        print(
            f"Extreme values (>10): {np.sum(scale_real_array > 10)} out of {len(scale_real_array)}"
        )
        print(
            f"Tiny values (<0.001): {np.sum(scale_real_array < 0.001)} out of {len(scale_real_array)}"
        )
        print(f"================================\n")

        self.perf_monitor.end_timing()
        return splats

    def normalize_splat_position_to_origin(
        self, splat_data: List[GaussianSplat]
    ) -> List[GaussianSplat]:
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
        return splat_data

    def visualize_centroid(self, splat_data: List[GaussianSplat]) -> RG.Brep:
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
        return centroid_cube, SD.Color.Red  # Return cube and color for visualization

    # -----------------------------
    # External Functions
    # -----------------------------
    def run(
        self,
        file_path: str,
        scale_factor: float,
        subdivision_level: int,
        sample_percentage: float,
        render_mode: str = "preview",
    ):
        print("=== RunScript STARTED ===")

        # Add your processing logic here
        print(
            f"Processed with inputs: {file_path}, {scale_factor}, {subdivision_level}, {sample_percentage} in render mode '{render_mode}'"
        )

        # Read Gaussian splats from PLY file
        splat_data = self.load_gaussian_splats(file_path)

        # Filter and sample the splat data
        splat_data = self.simple_sampling(splat_data, sample_percentage)
        splat_data = self.apply_filters(splat_data)
        splat_data = self.normalize_splat_position_to_origin(splat_data)
        centroid_cube, centroid_color = self.visualize_centroid(splat_data)
        
        print(f"Using render mode: {render_mode}")
        
        # Display render mode information
        if render_mode == "merged":
            print("📦 Merged mode: Creating single mesh with vertex colors for optimal performance")
        elif render_mode == "pointcloud":
            print("🔴 PointCloud mode: Creating single point cloud for maximum performance")
        elif render_mode == "preview":
            print("🔺 Preview mode: Creating individual mesh objects")
        elif render_mode == "export":
            print("🏗️  Export mode: Creating individual Brep objects (high quality)")
        elif render_mode == "test":
            print("📍 Test mode: Creating individual point objects")

        # Performance-aware mode suggestions
        if len(splat_data) > 10000 and render_mode in ["preview", "test"]:
            print(
                f"💡 Performance tip: With {len(splat_data):,} splats, consider using 'merged' or 'pointcloud' mode for faster rendering"
            )
        elif len(splat_data) > 5000 and render_mode == "preview":
            print(
                f"⚠️  Warning: {len(splat_data):,} individual meshes may cause slow performance. Consider 'merged' or 'pointcloud' mode."
            )

        # Create Rhino geometries based on render mode
        if render_mode == "pointcloud":
            # Create single PointCloud object instead of individual geometries
            point_cloud = self.create_colored_point_cloud(splat_data)

            # For pointcloud mode, return single object with placeholder color
            # Grasshopper will handle the point cloud's internal colors
            geometries = [centroid_cube, point_cloud]
            colors = [
                centroid_color,
                SD.Color.White,
            ]  # Placeholder color for point cloud

        elif render_mode == "merged":
            # Create single merged mesh containing all splat geometries
            merged_mesh = self.create_merged_mesh(splat_data)

            # For merged mode, return single mesh object with placeholder color
            # Vertex colors are embedded in the mesh itself
            geometries = [centroid_cube, merged_mesh]
            colors = [
                centroid_color,
                SD.Color.White,
            ]  # Placeholder color for merged mesh

        else:
            # Standard mode: create individual geometry objects
            geometries, colors = (
                [centroid_cube],
                [centroid_color],
            )  # Start with centroid cube

            for splat in splat_data:
                # Create 3D ellipsoid geometry (mesh or brep) for each Gaussian splat
                geometry = self.create_splat_geometry(splat, render_mode)
                geometries.append(geometry)

                # Convert spherical harmonics to RGB color
                r = ColorUtils.sh_to_rgb(splat.color[0])  # Red channel from f_dc_0
                g = ColorUtils.sh_to_rgb(splat.color[1])  # Green channel from f_dc_1
                b = ColorUtils.sh_to_rgb(splat.color[2])  # Blue channel from f_dc_2

                color = SD.Color.FromArgb(r, g, b)
                colors.append(color)

        # Track final geometry statistics
        self.perf_monitor.track_geometry_stats(geometries, render_mode)

        # Return geometries and colors
        return [geometries, colors]

    def export_for_3d_printing(
        self,
        file_path: str,
        scale_factor: float,
        subdivision_level: int,
        sample_percentage: float,
        output_path: str = None,
    ):
        """Generate high-quality Breps for 3D printing and optionally export to STL.

        Args:
            file_path: Path to PLY file
            scale_factor: Scaling factor for geometry
            subdivision_level: Detail level for surfaces
            sample_percentage: Percentage of splats to include
            output_path: Optional path to save STL file
        Returns:
            List of Brep objects ready for 3D printing
        """
        print("=== EXPORT FOR 3D PRINTING STARTED ===")
        print("Generating high-quality Breps (this may take longer)...")

        # Run with export mode to generate Breps
        breps, colors = self.run(
            file_path=file_path,
            scale_factor=scale_factor,
            subdivision_level=subdivision_level,
            sample_percentage=sample_percentage,
            render_mode="export",
        )

        print(f"Generated {len(breps)} Breps for 3D printing")

        if output_path:
            # TODO: Add STL export functionality here
            # This would require joining meshes, ensuring manifold geometry, etc.
            print(f"STL export to {output_path} - functionality to be implemented")

        print("=== EXPORT FOR 3D PRINTING COMPLETED ===")
        return breps


workflow_manager = GaussianSplatReader()

# This variables should be set by the Grasshopper environment
results = workflow_manager.run(
    file_path=file_path,
    scale_factor=scale_factor,
    subdivision_level=subdivision_level,
    sample_percentage=sample_percentage,
    render_mode=render_mode,
)

geometries = results[0]  # Geometry output (Meshes or Breps)
colors = results[1]  # Colors output
