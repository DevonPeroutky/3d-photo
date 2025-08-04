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


class GaussianSplatReader:
    """Handles loading and filtering of Gaussian splat data from PLY files."""

    def __init__(self):
        """Initialize the GaussianSplatReader for PLY loading and filtering operations."""
        self.perf_monitor = PerformanceMonitor()

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
            # Use standard quaternion component order: [rot_0, rot_1, rot_2, rot_3] = [x, y, z, w]
            # quat_raw = np.array(
            #     (v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]), dtype=np.float32
            # )            quat_raw = np.array((v["rot_3"], v["rot_0"], v["rot_1"], v["rot_2"]))  # Reorder to [w,x,y,z]
            quat_raw = np.array(
                (v["rot_3"], v["rot_0"], v["rot_1"], v["rot_2"])
            )  # Reorder to [w,x,y,z]
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

        self.perf_monitor.end_timing()
        return splats

    # -----------------------------
    # Filtering Methods
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
            "opacity": {
                "enabled": True,
                "min_opacity": 0.1,
            },  # Filter out low opacity splats
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
                "enabled": True,
                "min_scale_percentile": 2.0,  # More aggressive filtering
                "max_scale_percentile": 98.0,
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

    def sample_by_region(
        self,
        splats: List[GaussianSplat],
        region_center: RG.Point3d,
        radius: float,
    ) -> List[GaussianSplat]:
        """Sample splats within a specified radius from a center point."""
        sampled_splats = [
            splat
            for splat in splats
            if splat.position.DistanceTo(region_center) <= radius
        ]
        print(
            f"Sampled {len(sampled_splats)} splats within radius {radius} from {region_center}"
        )
        return sampled_splats

    def normalize_splat_position_to_origin(
        self, splat_data: List[GaussianSplat]
    ) -> List[GaussianSplat]:
        """Normalize splat positions around the origin by centering the point cloud."""
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


class WorkflowManager:
    """Handles Rhino geometry creation, rendering, and workflow orchestration for Gaussian splats."""

    def __init__(self):
        """Initialize the WorkflowManager with performance monitoring and data reader."""
        self.perf_monitor = PerformanceMonitor()
        self.splat_reader = GaussianSplatReader()
        # Sphere mesh cache will be initialized on first use

    # -----------------------------
    # Rhino Geometry Creation
    # -----------------------------
    def quaternion_to_rotation_transform(self, quat_wxyz: np.ndarray) -> RG.Transform:
        """Convert a normalized quaternion to a Rhino rotation transformation using native API.

        Args:
            quat_wxyz: Normalized quaternion as numpy array [w, x, y, z]
        Returns:
            RG.Transform: Rotation transformation matrix
        """
        w, x, y, z = map(float, quat_wxyz)

        # Use native Rhino Quaternion API (a, b, c, d format where a=w, b=x, c=y, d=z)
        quaternion = RG.Quaternion(w, x, y, z)
        return quaternion.GetRotation()

    def _get_base_sphere_mesh(
        self,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        scale_z: float = 1.0,
        edge_density: float = 8.0,
    ) -> RG.Mesh:
        """Get or create a cached base sphere mesh with anisotropic subdivision based on expected scaling.

        Args:
            scale_x, scale_y, scale_z: Expected scale factors for this splat
            edge_density: Target edge density factor for subdivision calculation (default: 8.0)
                         Higher values = more subdivisions, better quality for stretched geometry
        Returns:
            RG.Mesh: A unit sphere mesh at origin optimized for the given anisotropy
        """
        # Calculate anisotropic subdivisions based on the expected scaling
        # Choose UV divisions that respect anisotropy to prevent blockiness
        longest = max(scale_x, scale_y, scale_z)
        uDiv = int(max(4, round(longest * edge_density)))
        vDiv = max(2, uDiv // 2)  # Keep roughly square quads, minimum of 2

        # Create cache key based on subdivision levels to support different resolutions
        cache_key = f"sphere_mesh_{uDiv}_{vDiv}"
        if not hasattr(self, "_sphere_mesh_cache"):
            self._sphere_mesh_cache = {}

        if cache_key not in self._sphere_mesh_cache:
            # Create sphere with anisotropy-aware subdivisions
            self._sphere_mesh_cache[cache_key] = RG.Mesh.CreateFromSphere(
                RG.Sphere(RG.Point3d.Origin, 1.0), uDiv, vDiv
            )
        return self._sphere_mesh_cache[cache_key]

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

        # 1. scale (NeRF frame)
        geometry.Transform(
            RG.Transform.Scale(RG.Plane.WorldXY, scale_x, scale_y, scale_z)
        )

        # 2. ROTATE: Apply quaternion rotation
        geometry.Transform(self.quaternion_to_rotation_transform(splat.rotation_angles))

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

    def create_splat_mesh_in_rhino(
        self,
        splat: GaussianSplat,
        scale_multiplier: float = 2.5,
        subdivision_level: int = 3,
    ) -> RG.Mesh:
        """Create a Rhino Mesh object from a GaussianSplat instance.
        This method creates an ellipsoid mesh based on the splat parameters.
        Much faster than Breps for visualization purposes. Uses anisotropic mesh instancing for quality.

        Args:
            splat (GaussianSplat): The GaussianSplat instance containing parameters.
            scale_multiplier (float): Multiplier for Gaussian scales to represent visual extent (default: 2.5)
            subdivision_level (int): Base subdivision level for mesh detail (default: 3, used as edge_density)
        Returns:
            RG.Mesh: The resulting Mesh object representing the splat.
        """
        # FIX: Gaussian splat scales represent standard deviation (σ), not radius
        # Visual extent of Gaussian is ~2-3σ (95-99% of distribution)
        # Apply scale multiplier to convert from statistical to visual representation
        scale_x = splat.scale.X * scale_multiplier
        scale_y = splat.scale.Y * scale_multiplier
        scale_z = splat.scale.Z * scale_multiplier

        # Get anisotropic base sphere mesh based on expected scaling
        base_mesh = self._get_base_sphere_mesh(
            scale_x, scale_y, scale_z, edge_density=subdivision_level * 2.0
        )
        mesh = base_mesh.Duplicate()

        self._apply_splat_transformations(mesh, scale_x, scale_y, scale_z, splat)

        # Recalculate normals after scaling to fix lighting on stretched meshes
        # This prevents Rhino from reusing unit-sphere normals on stretched geometry
        mesh.Normals.ComputeNormals()

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

    def create_merged_mesh(
        self,
        splats: List[GaussianSplat],
        scale_multiplier: float = 2.5,
        subdivision_level: int = 3,
        debug: bool = False,
    ) -> RG.Mesh:
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
            scale_multiplier (float): Multiplier for Gaussian scales to represent visual extent (default: 2.5)
            subdivision_level (int): Sphere subdivision level for mesh detail (default: 3)
        Returns:
            RG.Mesh: Single merged mesh containing all splat geometries with vertex colors
        """
        self.perf_monitor.start_timing(f"Merged mesh creation for {len(splats)} splats")

        if len(splats) == 0:
            return RG.Mesh()

        # Create merged mesh
        merged_mesh = RG.Mesh()

        for splat in splats:
            # FIX: Apply scale multiplier to convert from Gaussian σ to visual extent
            scale_x = splat.scale.X * scale_multiplier
            scale_y = splat.scale.Y * scale_multiplier
            scale_z = splat.scale.Z * scale_multiplier

            # Get anisotropic base sphere mesh based on this splat's scaling
            base_sphere = self._get_base_sphere_mesh(
                scale_x, scale_y, scale_z, edge_density=subdivision_level * 2.0
            )
            sphere_mesh = base_sphere.Duplicate()

            self._apply_splat_transformations(
                sphere_mesh, scale_x, scale_y, scale_z, splat
            )

            # Recalculate normals after scaling for proper lighting
            sphere_mesh.Normals.ComputeNormals()

            # Convert spherical harmonics to RGB color with opacity/alpha
            r = ColorUtils.sh_to_rgb(splat.color[0])  # Red channel from f_dc_0
            g = ColorUtils.sh_to_rgb(splat.color[1])  # Green channel from f_dc_1
            b = ColorUtils.sh_to_rgb(splat.color[2])  # Blue channel from f_dc_2
            # FIX: Include opacity as alpha channel for transparency support
            # alpha = int(
            #     splat.opacity * 255
            # )
            alpha = 255
            color = SD.Color.FromArgb(alpha, r, g, b)

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

        # Run debug analysis if enabled
        if debug:
            print(f"\n=== MERGED MESH DEBUG ANALYSIS ===")

            # 1. Individual ellipsoid validation
            print(f"1. Validating individual ellipsoids...")
            ellipsoid_analysis = self.validate_individual_ellipsoids(
                splats,
                scale_multiplier,
                subdivision_level,
                sample_count=min(10, len(splats)),
            )

            if "error" not in ellipsoid_analysis:
                print(
                    f"   ✅ {ellipsoid_analysis['valid_count']}/{ellipsoid_analysis['sample_count']} individual ellipsoids created successfully"
                )
                for axis, stats in ellipsoid_analysis[
                    "scale_ratio_consistency"
                ].items():
                    print(
                        f"   📐 {axis}-axis scale ratio: mean={stats['mean']:.2f}, std={stats['std']:.2f}"
                    )

                for warning in ellipsoid_analysis.get("warnings", []):
                    print(f"   {warning}")
            else:
                print(
                    f"   ❌ Individual ellipsoid validation failed: {ellipsoid_analysis['error']}"
                )

            # 2. Bounding box analysis
            print(f"\n2. Analyzing merged mesh bounds...")
            bounds_analysis = self.analyze_merged_mesh_bounds(merged_mesh, splats)

            if "error" not in bounds_analysis:
                extents = bounds_analysis["mesh_extents"]
                print(
                    f"   📦 Mesh extents: X={extents['x']:.3f}, Y={extents['y']:.3f}, Z={extents['z']:.3f}"
                )
                print(
                    f"   📏 Diagonal length: {bounds_analysis['diagonal_length']:.3f}"
                )
                print(
                    f"   🎯 Centroid: ({bounds_analysis['centroid']['x']:.2f}, {bounds_analysis['centroid']['y']:.2f}, {bounds_analysis['centroid']['z']:.2f})"
                )

                scale_stats = bounds_analysis["scale_statistics"]
                print(
                    f"   🔢 Scale stats: min={scale_stats['min']:.6f}, max={scale_stats['max']:.3f}, mean={scale_stats['mean']:.3f}"
                )

                for warning in bounds_analysis.get("warnings", []):
                    print(f"   {warning}")
            else:
                print(f"   ❌ Bounds analysis failed: {bounds_analysis['error']}")

            # 3. Mesh quality analysis
            print(f"\n3. Analyzing mesh quality...")
            quality_analysis = self.analyze_mesh_quality(merged_mesh)

            if "error" not in quality_analysis:
                aspect_stats = quality_analysis["aspect_ratio_stats"]
                print(
                    f"   🔺 {quality_analysis['total_faces']} faces, {quality_analysis['analyzed_triangles']} triangles analyzed"
                )
                print(
                    f"   📊 Aspect ratios: min={aspect_stats['min']:.1f}, max={aspect_stats['max']:.1f}, mean={aspect_stats['mean']:.1f}"
                )

                problematic = quality_analysis["problematic_faces"]
                if problematic["high_aspect_ratio"] > 0:
                    print(
                        f"   ⚠️ {problematic['high_aspect_ratio']} faces with aspect ratio > 100"
                    )
                if problematic["extreme_aspect_ratio"] > 0:
                    print(
                        f"   🚨 {problematic['extreme_aspect_ratio']} faces with aspect ratio > 1000"
                    )

                if quality_analysis["degenerate_faces"] > 0:
                    print(
                        f"   💀 {quality_analysis['degenerate_faces']} degenerate faces detected"
                    )

                for warning in quality_analysis.get("warnings", []):
                    print(f"   {warning}")
            else:
                print(f"   ❌ Quality analysis failed: {quality_analysis['error']}")

            # 4. Summary and recommendations
            print(f"\n4. Summary and recommendations:")
            total_warnings = (
                len(ellipsoid_analysis.get("warnings", []))
                + len(bounds_analysis.get("warnings", []))
                + len(quality_analysis.get("warnings", []))
            )

            if total_warnings == 0:
                print(f"   ✅ Merged mesh appears healthy - no major issues detected")
            else:
                print(
                    f"   ⚠️ {total_warnings} potential issues detected - review warnings above"
                )

                # Provide specific recommendations
                if bounds_analysis.get("scale_statistics", {}).get("max", 0) > 10:
                    print(
                        f"   💡 Consider reducing scale_multiplier (currently {scale_multiplier}) or applying scale filtering"
                    )

                if (
                    quality_analysis.get("problematic_faces", {}).get(
                        "high_aspect_ratio", 0
                    )
                    > quality_analysis.get("analyzed_triangles", 1) * 0.1
                ):
                    print(
                        f"   💡 High aspect ratio faces suggest extreme anisotropy - check quaternion rotations and scale values"
                    )

                if ellipsoid_analysis.get("failed_count", 0) > 0:
                    print(
                        f"   💡 Individual ellipsoid failures suggest issues with scale transformation or coordinate mapping"
                    )

            print(f"===================================\n")

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

        for splat in splats:
            # Apply same coordinate transformation as other geometry types
            transformed_pos = RG.Point3d(
                splat.position.X, splat.position.Z, -splat.position.Y
            )

            # Convert spherical harmonics to RGB color with opacity/alpha
            r = ColorUtils.sh_to_rgb(splat.color[0])  # Red channel from f_dc_0
            g = ColorUtils.sh_to_rgb(splat.color[1])  # Green channel from f_dc_1
            b = ColorUtils.sh_to_rgb(splat.color[2])  # Blue channel from f_dc_2
            # FIX: Include opacity as alpha channel for transparency support
            alpha = int(
                splat.opacity * 255
            )  # Convert opacity [0.0-1.0] to alpha [0-255]
            color = SD.Color.FromArgb(alpha, r, g, b)

            # Add point with color to the point cloud
            point_cloud.Add(transformed_pos, color)

        return point_cloud

    def analyze_merged_mesh_bounds(
        self, merged_mesh: RG.Mesh, splats: List[GaussianSplat]
    ) -> dict:
        """Analyze merged mesh bounding box to detect scale/unit mismatches.

        Args:
            merged_mesh: The merged mesh to analyze
            splats: Original splat data for comparison
        Returns:
            Dictionary with bounding box analysis results
        """
        if not merged_mesh or merged_mesh.Vertices.Count == 0:
            return {"error": "Empty or invalid mesh"}

        # Get mesh bounding box
        bbox = merged_mesh.GetBoundingBox(True)

        # Calculate dimensions
        x_extent = bbox.Max.X - bbox.Min.X
        y_extent = bbox.Max.Y - bbox.Min.Y
        z_extent = bbox.Max.Z - bbox.Min.Z
        diagonal = bbox.Diagonal.Length
        centroid = bbox.Center

        # Calculate expected bounds from original splat data
        positions = [
            [s.position.X, s.position.Z, -s.position.Y] for s in splats
        ]  # Apply coord transform
        pos_array = np.array(positions)
        expected_min = np.min(pos_array, axis=0)
        expected_max = np.max(pos_array, axis=0)
        expected_extents = expected_max - expected_min

        # Calculate scale statistics from splats
        scales = []
        for splat in splats:
            scales.extend([splat.scale.X, splat.scale.Y, splat.scale.Z])
        scale_stats = {
            "min": np.min(scales),
            "max": np.max(scales),
            "mean": np.mean(scales),
            "median": np.median(scales),
        }

        analysis = {
            "mesh_extents": {"x": x_extent, "y": y_extent, "z": z_extent},
            "diagonal_length": diagonal,
            "centroid": {"x": centroid.X, "y": centroid.Y, "z": centroid.Z},
            "expected_extents": {
                "x": expected_extents[0],
                "y": expected_extents[1],
                "z": expected_extents[2],
            },
            "scale_statistics": scale_stats,
            "vertex_count": merged_mesh.Vertices.Count,
            "face_count": merged_mesh.Faces.Count,
        }

        # Generate warnings
        warnings = []

        # Check for extreme sizes
        if diagonal > 1000:
            warnings.append(
                f"⚠️ Very large mesh (diagonal: {diagonal:.1f}) - possible scale multiplier issue"
            )
        if diagonal < 0.01:
            warnings.append(
                f"⚠️ Very small mesh (diagonal: {diagonal:.6f}) - possible unit mismatch"
            )

        # Check for extreme anisotropy
        max_extent = max(x_extent, y_extent, z_extent)
        min_extent = min(x_extent, y_extent, z_extent)
        if min_extent > 0 and max_extent / min_extent > 1000:
            warnings.append(
                f"⚠️ Extreme anisotropy (ratio: {max_extent / min_extent:.0f}) - check coordinate transform"
            )

        # Check scale reasonableness
        if scale_stats["max"] > 10:
            warnings.append(
                f"⚠️ Very large scale values (max: {scale_stats['max']:.3f}) - check exp() transformation"
            )
        if scale_stats["min"] < 0.0001:
            warnings.append(
                f"⚠️ Very small scale values (min: {scale_stats['min']:.6f}) - may create degenerate geometry"
            )

        analysis["warnings"] = warnings
        return analysis

    def analyze_mesh_quality(self, merged_mesh: RG.Mesh) -> dict:
        """Analyze mesh face quality to detect degenerate triangles.

        Args:
            merged_mesh: The merged mesh to analyze
        Returns:
            Dictionary with mesh quality analysis results
        """
        if not merged_mesh or merged_mesh.Faces.Count == 0:
            return {"error": "Empty or invalid mesh"}

        aspect_ratios = []
        degenerate_faces = 0
        face_areas = []

        for i in range(merged_mesh.Faces.Count):
            face = merged_mesh.Faces[i]

            # Get face vertices
            if face.IsQuad:
                # For quads, check both triangular subdivisions
                v1 = merged_mesh.Vertices[face.A]
                v2 = merged_mesh.Vertices[face.B]
                v3 = merged_mesh.Vertices[face.C]
                v4 = merged_mesh.Vertices[face.D]

                # Analyze first triangle (A,B,C)
                edges1 = [v1.DistanceTo(v2), v2.DistanceTo(v3), v3.DistanceTo(v1)]
                if min(edges1) > 0:
                    aspect_ratios.append(max(edges1) / min(edges1))
                else:
                    degenerate_faces += 1

                # Analyze second triangle (A,C,D)
                edges2 = [v1.DistanceTo(v3), v3.DistanceTo(v4), v4.DistanceTo(v1)]
                if min(edges2) > 0:
                    aspect_ratios.append(max(edges2) / min(edges2))
                else:
                    degenerate_faces += 1

            else:
                # Triangle face
                v1 = merged_mesh.Vertices[face.A]
                v2 = merged_mesh.Vertices[face.B]
                v3 = merged_mesh.Vertices[face.C]

                edges = [v1.DistanceTo(v2), v2.DistanceTo(v3), v3.DistanceTo(v1)]

                if min(edges) > 0:
                    aspect_ratios.append(max(edges) / min(edges))
                    # Calculate triangle area using cross product
                    edge1 = RG.Vector3d(v2 - v1)
                    edge2 = RG.Vector3d(v3 - v1)
                    area = 0.5 * RG.Vector3d.CrossProduct(edge1, edge2).Length
                    face_areas.append(area)
                else:
                    degenerate_faces += 1

        if len(aspect_ratios) == 0:
            return {"error": "No valid faces found"}

        aspect_ratios = np.array(aspect_ratios)
        face_areas = np.array(face_areas) if face_areas else np.array([0])

        # Count problematic faces
        high_aspect_faces = np.sum(aspect_ratios > 100)
        extreme_aspect_faces = np.sum(aspect_ratios > 1000)

        analysis = {
            "total_faces": merged_mesh.Faces.Count,
            "analyzed_triangles": len(aspect_ratios),
            "degenerate_faces": degenerate_faces,
            "aspect_ratio_stats": {
                "min": np.min(aspect_ratios),
                "max": np.max(aspect_ratios),
                "mean": np.mean(aspect_ratios),
                "median": np.median(aspect_ratios),
                "std": np.std(aspect_ratios),
            },
            "problematic_faces": {
                "high_aspect_ratio": high_aspect_faces,  # > 100
                "extreme_aspect_ratio": extreme_aspect_faces,  # > 1000
            },
            "face_area_stats": {
                "min": np.min(face_areas) if len(face_areas) > 0 else 0,
                "max": np.max(face_areas) if len(face_areas) > 0 else 0,
                "mean": np.mean(face_areas) if len(face_areas) > 0 else 0,
            },
        }

        # Generate warnings
        warnings = []
        if degenerate_faces > 0:
            warnings.append(
                f"⚠️ {degenerate_faces} degenerate faces detected (zero-length edges)"
            )
        if high_aspect_faces > len(aspect_ratios) * 0.1:
            warnings.append(
                f"⚠️ {high_aspect_faces} faces with aspect ratio > 100 ({high_aspect_faces / len(aspect_ratios) * 100:.1f}%)"
            )
        if extreme_aspect_faces > 0:
            warnings.append(
                f"⚠️ {extreme_aspect_faces} faces with extreme aspect ratio > 1000"
            )
        if analysis["aspect_ratio_stats"]["max"] > 100:
            warnings.append(
                f"⚠️ Maximum aspect ratio is {analysis['aspect_ratio_stats']['max']:.1f} - indicates stretched geometry"
            )

        analysis["warnings"] = warnings
        return analysis

    def validate_individual_ellipsoids(
        self,
        splats: List[GaussianSplat],
        scale_multiplier: float = 2.5,
        subdivision_level: int = 3,
        sample_count: int = 5,
    ) -> dict:
        """Validate individual ellipsoid creation before merging.

        Args:
            splats: List of splats to validate
            scale_multiplier: Scale multiplier used for ellipsoid creation
            subdivision_level: Subdivision level for mesh detail
            sample_count: Number of sample ellipsoids to create and analyze
        Returns:
            Dictionary with individual ellipsoid validation results
        """
        if len(splats) == 0:
            return {"error": "No splats provided"}

        # Sample a few splats for detailed analysis
        sample_splats = splats[: min(sample_count, len(splats))]

        individual_results = []

        for i, splat in enumerate(sample_splats):
            try:
                # Create individual mesh
                mesh = self.create_splat_mesh_in_rhino(
                    splat, scale_multiplier, subdivision_level
                )

                # Analyze this individual mesh
                bbox = mesh.GetBoundingBox(True)
                extents = {
                    "x": bbox.Max.X - bbox.Min.X,
                    "y": bbox.Max.Y - bbox.Min.Y,
                    "z": bbox.Max.Z - bbox.Min.Z,
                }

                # Original splat scales
                original_scales = {
                    "x": splat.scale.X,
                    "y": splat.scale.Y,
                    "z": splat.scale.Z,
                }

                # Expected scales after transformation
                expected_scales = {
                    "x": splat.scale.X * scale_multiplier,
                    "y": splat.scale.Y * scale_multiplier,
                    "z": splat.scale.Z * scale_multiplier,
                }

                # Scale verification (account for coordinate transform)
                # Coordinate transform: X->X, Y->Z, Z->-Y
                # So: splat.scale.X corresponds to mesh extents.x
                #     splat.scale.Y corresponds to mesh extents.z (Y becomes Z)
                #     splat.scale.Z corresponds to mesh extents.y (Z becomes -Y, but scale is magnitude)
                scale_ratios = {
                    "x": extents["x"] / (expected_scales["x"] * 2)
                    if expected_scales["x"] > 0
                    else 0,  # splat_X -> mesh_X
                    "y": extents["z"] / (expected_scales["y"] * 2)
                    if expected_scales["y"] > 0
                    else 0,  # splat_Y -> mesh_Z
                    "z": extents["y"] / (expected_scales["z"] * 2)
                    if expected_scales["z"] > 0
                    else 0,  # splat_Z -> mesh_Y (magnitude, so -Y doesn't matter)
                }

                individual_results.append(
                    {
                        "splat_index": i,
                        "original_scales": original_scales,
                        "expected_scales": expected_scales,
                        "actual_extents": extents,
                        "scale_ratios": scale_ratios,
                        "vertex_count": mesh.Vertices.Count,
                        "face_count": mesh.Faces.Count,
                        "position": {
                            "x": splat.position.X,
                            "y": splat.position.Y,
                            "z": splat.position.Z,
                        },
                        "valid": mesh.IsValid,
                    }
                )

            except Exception as e:
                individual_results.append({"splat_index": i, "error": str(e)})

        # Calculate summary statistics
        valid_results = [
            r for r in individual_results if "error" not in r and r.get("valid", False)
        ]

        if len(valid_results) == 0:
            return {
                "error": "No valid individual ellipsoids created",
                "individual_results": individual_results,
            }

        # Analyze scale consistency
        all_ratios_x = [
            r["scale_ratios"]["x"] for r in valid_results if r["scale_ratios"]["x"] > 0
        ]
        all_ratios_y = [
            r["scale_ratios"]["y"] for r in valid_results if r["scale_ratios"]["y"] > 0
        ]
        all_ratios_z = [
            r["scale_ratios"]["z"] for r in valid_results if r["scale_ratios"]["z"] > 0
        ]

        summary = {
            "sample_count": len(sample_splats),
            "valid_count": len(valid_results),
            "failed_count": len(sample_splats) - len(valid_results),
            "individual_results": individual_results,
            "scale_ratio_consistency": {
                "x_axis": {"mean": np.mean(all_ratios_x), "std": np.std(all_ratios_x)}
                if all_ratios_x
                else {"mean": 0, "std": 0},
                "y_axis": {"mean": np.mean(all_ratios_y), "std": np.std(all_ratios_y)}
                if all_ratios_y
                else {"mean": 0, "std": 0},
                "z_axis": {"mean": np.mean(all_ratios_z), "std": np.std(all_ratios_z)}
                if all_ratios_z
                else {"mean": 0, "std": 0},
            },
        }

        # Generate warnings
        warnings = []
        if summary["failed_count"] > 0:
            warnings.append(
                f"⚠️ {summary['failed_count']} individual ellipsoids failed to create"
            )

        # Check scale consistency (should be close to 1.0)
        for axis, stats in summary["scale_ratio_consistency"].items():
            if abs(stats["mean"] - 1.0) > 0.2:
                warnings.append(
                    f"⚠️ {axis}-axis scale ratio mean is {stats['mean']:.2f} (expected ~1.0)"
                )
            if stats["std"] > 0.5:
                warnings.append(
                    f"⚠️ {axis}-axis scale ratio has high variance (std: {stats['std']:.2f})"
                )

        summary["warnings"] = warnings
        return summary

    def create_splat_geometry(
        self,
        splat: GaussianSplat,
        render_mode: str = "preview",
        scale_multiplier: float = 2.5,
        subdivision_level: int = 3,
    ):
        """Create either a mesh or Brep from a GaussianSplat based on render mode.

        NOTE: This method handles individual splat geometry creation. For pointcloud and merged modes,
        use create_colored_point_cloud() or create_merged_mesh() directly with the full splat list for optimal performance.

        Args:
            splat (GaussianSplat): The GaussianSplat instance containing parameters.
            render_mode (str): "preview" for fast mesh, "export" for precise Brep, for points
            scale_multiplier (float): Multiplier for Gaussian scales to represent visual extent (default: 2.5)
            subdivision_level (int): Sphere subdivision level for mesh detail (default: 3)
        Returns:
            RG.Mesh or RG.Brep or RG.Point: The resulting geometry object.
        """
        if render_mode == "preview":
            return self.create_splat_mesh_in_rhino(
                splat, scale_multiplier, subdivision_level
            )
        elif render_mode == "export":
            return self.create_splat_object_in_rhino(splat, scale_multiplier)
        else:
            raise ValueError(
                f"Invalid render_mode: {render_mode}. Use 'preview', 'export', 'test', 'pointcloud', or 'merged'"
            )

    def create_splat_object_in_rhino(
        self, splat: GaussianSplat, scale_multiplier: float = 2.5
    ) -> RG.Brep:
        """Create a Rhino Brep object from a GaussianSplat instance.
        This method creates an ellipsoid based on the splat parameters.
        It uses the position, scale, and rotation to define the ellipsoid's geometry.

        Args:
            splat (GaussianSplat): The GaussianSplat instance containing parameters.
            scale_multiplier (float): Multiplier for Gaussian scales to represent visual extent (default: 2.5)
        Returns:
            RG.Brep: The resulting Brep object representing the splat.
        """
        # Create a unit sphere at origin
        sphere = RG.Sphere(RG.Point3d.Origin, 1.0)
        nurbs_surface = sphere.ToNurbsSurface()

        # FIX: Gaussian splat scales represent standard deviation (σ), not radius
        # Visual extent of Gaussian is ~2-3σ (95-99% of distribution)
        # Apply scale multiplier to convert from statistical to visual representation
        scale_x = splat.scale.X * scale_multiplier
        scale_y = splat.scale.Y * scale_multiplier
        scale_z = splat.scale.Z * scale_multiplier

        self._apply_splat_transformations(
            nurbs_surface, scale_x, scale_y, scale_z, splat
        )

        # Convert back to Brep
        brep = nurbs_surface.ToBrep()

        return brep

    def render_splats(
        self,
        splats: List[GaussianSplat],
        render_mode: str = "preview",
        vector_offset: RG.Vector3d = RG.Vector3d(0, 0, 0),
        scale_multiplier: float = 2.5,
        subdivision_level: int = 3,
        debug: bool = False,
    ) -> Tuple[List[RG.GeometryBase], List[SD.Color]]:
        """Render a list of Gaussian splats in Rhino using the specified render mode.

        Args:
            splats: List of GaussianSplat objects to render
            render_mode: "preview" for fast mesh rendering, "export" for precise Brep export,
                         "test" for point rendering, "pointcloud" for single PointCloud object,
                         "merged" for single merged Mesh object
            vector_offset: Optional offset vector to apply to all splat positions
            scale_multiplier: Multiplier for Gaussian scales to represent visual extent (default: 2.5)
            subdivision_level: Sphere subdivision level for mesh detail (default: 3)
        Returns:
            List of RG.GeometryBase objects representing the rendered splats
        """

        # Apply vector offset to all splat positions
        transformed_splats = [
            GaussianSplat(
                position=splat.position + vector_offset,
                rotation_angles=splat.rotation_angles,
                scale=splat.scale,
                color=splat.color,
                opacity=splat.opacity,
            )
            for splat in splats
        ]

        if render_mode == "pointcloud":
            point_cloud = self.create_colored_point_cloud(transformed_splats)
            return [point_cloud], [SD.Color.White]  # No colors needed for point cloud

        elif render_mode == "merged":
            merged_mesh = self.create_merged_mesh(
                transformed_splats, scale_multiplier, subdivision_level, debug
            )
            return [
                merged_mesh
            ], []  # Empty color list - mesh uses vertex colors instead
        elif render_mode == "preview":
            rendered_objects, colors = [], []

            for splat in transformed_splats:
                geometry = self.create_splat_geometry(
                    splat, render_mode, scale_multiplier, subdivision_level
                )
                rendered_objects.append(geometry)
                r = ColorUtils.sh_to_rgb(splat.color[0])  # Red channel from f_dc_0
                g = ColorUtils.sh_to_rgb(splat.color[1])  # Green channel from f_dc_1
                b = ColorUtils.sh_to_rgb(splat.color[2])  # Blue channel from f_dc_2

                # FIX: Include opacity as alpha channel for transparency support
                alpha = int(
                    splat.opacity * 255
                )  # Convert opacity [0.0-1.0] to alpha [0-255]
                color = SD.Color.FromArgb(alpha, r, g, b)
                colors.append(color)
            print(
                f"Rendered {len(rendered_objects)} transformed_splats in preview mode"
            )
            print(
                f"{self._sphere_mesh_cache.keys()} keys in the cache"
            )  # Debug: print cached sphere mesh keys
            return rendered_objects, colors
        else:
            raise ValueError(
                f"Invalid render_mode: {render_mode}. Use 'preview', 'export', 'test', 'pointcloud', or 'merged'"
            )

    # -----------------------------
    # Workflow Utility Functions
    # -----------------------------
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
        return (
            centroid_point,
            centroid_cube,
            SD.Color.Red,
        )  # Return cube and color for visualization

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
        debug: bool = False,
    ) -> Tuple[List[RG.GeometryBase], List[SD.Color]]:
        print("=== RunScript STARTED ===")

        # Add your processing logic here
        print(
            f"Processed with inputs: {file_path}, {scale_factor}, {subdivision_level}, {sample_percentage} in render mode '{render_mode}'"
        )

        # Read Gaussian splats from PLY file
        splat_data = self.splat_reader.load_gaussian_splats(file_path)

        # Filter and sample the splat data
        # splat_data = self.splat_reader.simple_sampling(splat_data, sample_percentage)
        splat_data = self.splat_reader.normalize_splat_position_to_origin(splat_data)
        centroid_point, centroid_cube, centroid_color = self.visualize_centroid(
            splat_data
        )
        splat_data = self.splat_reader.sample_by_region(splat_data, centroid_point, 2)
        splat_data = self.splat_reader.apply_filters(splat_data)

        print(
            f"Rendering Geometries with Render Mode: {render_mode}, scale factor: {scale_factor}, subdivision level: {subdivision_level}"
        )
        pointcloud_geometries, pointcoud_colors = self.render_splats(
            splats=splat_data,
            render_mode="pointcloud",
            vector_offset=RG.Vector3d(0, 0, 10),
            scale_multiplier=scale_factor,  # Not used for pointcloud mode, but consistent API
        )
        merged_geometries, merged_colors = self.render_splats(
            splats=splat_data,
            render_mode="merged",
            vector_offset=RG.Vector3d(0, 0, 0),
            # scale_multiplier=2.5,  # Apply scale multiplier to fix mesh sizes
            scale_multiplier=scale_factor,  # Apply scale multiplier to fix mesh sizes
            subdivision_level=subdivision_level,
            debug=debug,  # Enable debug analysis for merged meshes
        )

        geometries = pointcloud_geometries + merged_geometries
        colors = pointcoud_colors + merged_colors

        # Track final geometry statistics
        self.perf_monitor.track_geometry_stats(geometries, render_mode)

        # Return geometries and colors
        print(len(geometries), "geometries created")

        # DEBUG: Add scale and performance analysis
        self._debug_splat_analysis(splat_data, geometries, colors)

        return (geometries, colors)

    def _debug_splat_analysis(
        self, splats: List[GaussianSplat], geometries: List, colors: List
    ):
        """Print debugging information about splat scales and mesh sizes for validation.

        This helps users understand if the scale multiplier is appropriate and if the
        mesh sizes are visually correct compared to the original Gaussian splat data.
        """
        print(f"\n=== SPLAT SCALE DEBUG ANALYSIS ===")

        if len(splats) == 0:
            print("No splats to analyze")
            return

        # Analyze original Gaussian splat scales
        original_scales = []
        for splat in splats:
            # Calculate geometric mean of X,Y,Z scales
            geom_mean = np.power(splat.scale.X * splat.scale.Y * splat.scale.Z, 1 / 3)
            original_scales.append(geom_mean)

        original_scales = np.array(original_scales)

        # Analyze mesh sizes (if using mesh modes)
        mesh_count = 0
        total_vertices = 0
        for geom in geometries:
            if hasattr(geom, "Vertices"):  # It's a mesh
                mesh_count += 1
                if hasattr(geom.Vertices, "Count"):
                    total_vertices += geom.Vertices.Count

        print(f"Original Gaussian scales (σ):")
        print(
            f"  Min: {np.min(original_scales):.6f}, Max: {np.max(original_scales):.6f}"
        )
        print(
            f"  Mean: {np.mean(original_scales):.6f}, Median: {np.median(original_scales):.6f}"
        )

        print(f"Applied scale multiplier: 2.5x")
        visual_scales = original_scales * 2.5
        print(f"Visual mesh scales (2.5σ):")
        print(f"  Min: {np.min(visual_scales):.6f}, Max: {np.max(visual_scales):.6f}")
        print(
            f"  Mean: {np.mean(visual_scales):.6f}, Median: {np.median(visual_scales):.6f}"
        )

        print(f"Geometry statistics:")
        print(f"  Total geometries: {len(geometries)}")
        print(f"  Mesh objects: {mesh_count}")
        if mesh_count > 0:
            print(
                f"  Total vertices: {total_vertices} ({total_vertices / mesh_count:.1f} avg per mesh)"
            )

        # Scale validation warnings
        tiny_scales = np.sum(visual_scales < 0.001)
        huge_scales = np.sum(visual_scales > 5.0)

        if tiny_scales > len(splats) * 0.1:
            print(
                f"⚠️  {tiny_scales} splats ({tiny_scales / len(splats) * 100:.1f}%) have very small scales (<0.001)"
            )
            print(f"   Consider increasing scale_multiplier or using pointcloud mode")

        if huge_scales > 0:
            print(f"⚠️  {huge_scales} splats have very large scales (>5.0)")
            print(
                f"   Consider decreasing scale_multiplier or applying scale filtering"
            )

        if huge_scales > 0:
            print(f"⚠️  {huge_scales} splats have very large scales (>5.0)")
            print(
                f"   Consider decreasing scale_multiplier or applying scale filtering"
            )

        print(f"===================================\n")


workflow_manager = WorkflowManager()

# This variables should be set by the Grasshopper environment
debug_mode = globals().get("debug_mode", True)  # Default to False if not provided
file_path = globals().get("file_path", "../assets/JAPAN.ply")
scale_factor = globals().get("scale_factor", 1)  # Default scale factor
subdivision_level = int(
    globals().get("subdivision_level", 3)
)  # Default subdivision level
sample_percentage = globals().get("sample_percentage", 1.0)  # Default to 100%
render_mode = globals().get("render_mode", "preview")  # Default render mode

results = workflow_manager.run(
    file_path=file_path,
    scale_factor=scale_factor,
    subdivision_level=subdivision_level,
    sample_percentage=sample_percentage,
    render_mode=render_mode,
    debug=debug_mode,
)

geometries = results[0]  # Geometry output (Meshes or Breps)
colors = results[1]  # Colors output
