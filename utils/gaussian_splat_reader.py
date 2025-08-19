#! python 3
"""
Gaussian Splat Reader Module

Handles loading and filtering of Gaussian splat data from PLY files.
Provides comprehensive filtering options for noise reduction and data quality improvement.
"""

import plyfile
import random
import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any
from numpy.typing import NDArray
from enum import Enum

import Rhino.Geometry as RG
from performance_monitor import PerformanceMonitor
from custom_types import GaussianSplat, QuatFormat

DEFAULT_FILTER_CONFIG = {
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


class GaussianSplatReader:
    """Handles loading and filtering of Gaussian splat data from PLY files."""

    def __init__(self):
        """Initialize the GaussianSplatReader for PLY loading and filtering operations."""
        self.perf_monitor = PerformanceMonitor()

    def normlize_quaternions(
        self, vertex, format: QuatFormat, eps: float = 1e-12
    ) -> NDArray[np.float64]:
        """Normalize quaternion values to ensure they have unit length.
        
        Always returns quaternion in WXYZ format [w, x, y, z] for compatibility
        with quaternion_to_rotation_transform_custom function.
        
        Args:
            vertex: PLY vertex data containing rot_0, rot_1, rot_2, rot_3 fields
            format: Expected input format (WXYZ or XYZW)
            eps: Minimum norm threshold for valid quaternions
            
        Returns:
            Normalized quaternion as [w, x, y, z] in WXYZ format
        """
        # Extract quaternion components based on input format
        # Most PLY files store quaternions as [w, x, y, z] in rot_0, rot_1, rot_2, rot_3
        if format == QuatFormat.WXYZ:
            # PLY fields rot_0=w, rot_1=x, rot_2=y, rot_3=z
            quat_raw = np.array(
                [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]],
                dtype=np.float64,
            )
        else:  # QuatFormat.XYZW
            # PLY fields rot_0=x, rot_1=y, rot_2=z, rot_3=w -> reorder to [w,x,y,z]
            quat_raw = np.array(
                [vertex["rot_3"], vertex["rot_0"], vertex["rot_1"], vertex["rot_2"]],
                dtype=np.float64,
            )
        
        # Validate quaternion components
        assert len(quat_raw) == 4, "Quaternion must have exactly 4 components"
        
        # Calculate quaternion magnitude
        quat_norm = np.linalg.norm(quat_raw)
        
        # Handle degenerate quaternions
        fallback_rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # Identity quaternion [w,x,y,z]
        if quat_norm <= eps:
            return fallback_rotation
            
        # Normalize to unit quaternion
        quat_normalized = quat_raw / quat_norm
        
        # Ensure canonical form: W component should be non-negative
        # If W < 0, flip entire quaternion (q and -q represent same rotation)
        w_component = quat_normalized[0]  # W is always first element in our WXYZ format
        if w_component < 0:
            quat_normalized = -quat_normalized
            
        # Final validation: ensure we're returning WXYZ format
        assert quat_normalized.shape == (4,), "Output must be 4-element quaternion"
        
        return quat_normalized

    def load_gaussian_splats(
        self, file_path: str, quat_format: QuatFormat = QuatFormat.WXYZ
    ) -> List[GaussianSplat]:
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
                    rotation_angles=self.normlize_quaternions(
                        v, quat_format
                    ),  # Now using normalized quaternions
                    color=np.array(
                        (v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]), dtype=np.float32
                    ),
                    opacity=opacity_real,  # Now using properly transformed opacity
                    quat_format=quat_format,
                    normal=np.array((v["nx"], v["ny"], v["nz"]), dtype=np.float64),
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

    def apply_filters(
        self,
        splats: List[GaussianSplat],
        filter_config: Dict[str, Any] = DEFAULT_FILTER_CONFIG,
    ) -> List[GaussianSplat]:
        """Apply multiple filters in sequence based on configuration.

        Args:
            splats: List of GaussianSplat objects
            filter_config: Dictionary specifying which filters to apply and their parameters
        Returns:
            Filtered list of splats
        """
        self.perf_monitor.start_timing("Filtering operations")

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
                quat_format=splat.quat_format,
                normal=splat.normal,
            )
            for splat in splat_data
        ]
        return splat_data
