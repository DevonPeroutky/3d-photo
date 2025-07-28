"""
Color utilities for converting between different color spaces and formats.
"""

import math
import numpy as np
from typing import Tuple, List


class ColorUtils:
    """Utility class for color space conversions and transformations."""

    @staticmethod
    def sh_to_rgb(sh_coeff: float) -> int:
        """
        Convert spherical harmonics DC coefficients to RGB color values.
        This function applies a sigmoid transformation to the spherical harmonics
        from unbounded real values to a [0,1] range, then scales to [0,255].
        
        Args:
            sh_coeff: Spherical harmonics coefficient (unbounded real number)
        Returns:
            RGB color value as integer [0,255]
        """
        # Apply sigmoid activation: 1 / (1 + exp(-sh_coeff))
        # This converts from spherical harmonics space to probability space [0,1]
        sigmoid = 1.0 / (1.0 + math.exp(-sh_coeff))
        # Scale from [0,1] to [0,255] range for RGB color values
        return int(sigmoid * 255)

    @staticmethod
    def sh_to_rgb_normalized(sh_coeff: float) -> float:
        """
        Convert spherical harmonics DC coefficients to normalized RGB [0,1].
        
        Args:
            sh_coeff: Spherical harmonics coefficient (unbounded real number)
        Returns:
            RGB color value as float [0,1]
        """
        # Apply sigmoid activation: 1 / (1 + exp(-sh_coeff))
        return 1.0 / (1.0 + math.exp(-sh_coeff))

    @staticmethod
    def sh_array_to_rgb(sh_coeffs: np.ndarray) -> Tuple[int, int, int]:
        """
        Convert array of 3 spherical harmonics coefficients to RGB tuple.
        
        Args:
            sh_coeffs: Array of 3 SH coefficients [f_dc_0, f_dc_1, f_dc_2]
        Returns:
            RGB tuple as integers [0,255]
        """
        r = ColorUtils.sh_to_rgb(sh_coeffs[0])
        g = ColorUtils.sh_to_rgb(sh_coeffs[1])
        b = ColorUtils.sh_to_rgb(sh_coeffs[2])
        return (r, g, b)

    @staticmethod
    def sh_array_to_rgb_normalized(sh_coeffs: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert array of 3 spherical harmonics coefficients to normalized RGB tuple.
        
        Args:
            sh_coeffs: Array of 3 SH coefficients [f_dc_0, f_dc_1, f_dc_2]
        Returns:
            RGB tuple as floats [0,1]
        """
        r = ColorUtils.sh_to_rgb_normalized(sh_coeffs[0])
        g = ColorUtils.sh_to_rgb_normalized(sh_coeffs[1])
        b = ColorUtils.sh_to_rgb_normalized(sh_coeffs[2])
        return (r, g, b)

    @staticmethod
    def clamp_rgb(r: int, g: int, b: int) -> Tuple[int, int, int]:
        """
        Clamp RGB values to valid [0,255] range.
        
        Args:
            r, g, b: RGB values that may be outside valid range
        Returns:
            Clamped RGB tuple
        """
        return (
            max(0, min(255, r)),
            max(0, min(255, g)),
            max(0, min(255, b))
        )