from typing import List
import math
import Rhino.Geometry as RG

from gaussian_splat import GaussianSplat


def create_example_splats() -> List[GaussianSplat]:
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
