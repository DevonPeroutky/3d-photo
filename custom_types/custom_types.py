from enum import Enum
import Rhino.Geometry as RG
import numpy as np
from typing import Optional
from numpy.typing import NDArray


class QuatFormat(Enum):
    WXYZ = "wxyz"  # Standard quaternion format: [w, x, y, z]
    XYZW = "xyzw"  # Alternative format: [x, y, z, w] (less common)


class GaussianSplat:
    """Simple data structure to hold Gaussian splat parameters"""

    def __init__(
        self,
        position: RG.Point3d,
        scale: RG.Vector3d,
        rotation_angles: NDArray[np.float64],
        color: NDArray[np.float32],
        opacity: float,
        quat_format: QuatFormat,
        normal: Optional[NDArray[np.float64]] = None,
    ):
        self.position = position  # Point3d - center position
        self.scale = scale  # Vector3d - radii in X,Y,Z directions
        self.rotation_angles = (
            rotation_angles  # tuple (w, x, y, z) quaternion components
        )
        self.color = color  # tuple (R,G,B) 0-255
        self.opacity = opacity  # float 0.0-1.0
        self.normal = normal
        self.quat_format = quat_format

    def __str__(self):
        return (
            f"GaussianSplat(position={self.position}, scale={self.scale}, "
            f"rotation_angles({self.quat_format})={self.rotation_angles}, color={self.color}, "
            f"opacity={self.opacity})"
        )

    def convert_quaternion_format(self, format: QuatFormat):
        """Convert quaternion format if necessary"""
        if self.quat_format == format:
            return
        if format == QuatFormat.WXYZ and self.quat_format == QuatFormat.XYZW:
            # Convert from XYZW to WXYZ
            (x, y, z, w) = self.rotation_angles
            self.rotation_angles = np.array([w, x, y, z])
        elif format == QuatFormat.XYZW and self.quat_format == QuatFormat.WXYZ:
            # Convert from WXYZ to XYZW
            (w, x, y, z) = self.rotation_angles
            self.rotation_angles = np.array([x, y, z, w])
        self.quat_format = format
