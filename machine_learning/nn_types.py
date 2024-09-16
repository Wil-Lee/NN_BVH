from typing import Tuple
from enum import Enum
import numpy as np


Vertex3 = np.ndarray
""" 3D Vertex """

Primitive3 = np.ndarray
""" 3D Triangle Primitive"""

Mesh3 = list[Primitive3]
""" 3D Triangle Mesh """

class Axis(Enum):
    x = 0
    y = 1
    z = 2