import Rhino.Geometry as RG
import numpy as np
from typing import Optional
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
