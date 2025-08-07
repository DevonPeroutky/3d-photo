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
