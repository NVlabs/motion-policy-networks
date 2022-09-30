# MIT License
#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, University of Washington. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from dataclasses import dataclass, field
from typing import List, Union, Optional, Dict, Sequence
import numpy as np
from geometrout.transform import SE3
from geometrout.primitive import Cuboid, Cylinder, Sphere


Obstacles = List[Union[Cuboid, Cylinder, Sphere]]
Trajectory = Sequence[Union[Sequence, np.ndarray]]


@dataclass
class PlanningProblem:
    """
    Defines a common interface to describe planning problems
    """

    target: SE3  # The target in the `right_gripper` frame
    target_volume: Union[Cuboid, Cylinder]
    q0: np.ndarray  # The starting configuration
    obstacles: Optional[Obstacles] = None  # The obstacles in the scene
    obstacle_point_cloud: Optional[np.ndarray] = None
    target_negative_volumes: Obstacles = field(default_factory=lambda: [])


ProblemSet = Dict[str, Dict[str, List[PlanningProblem]]]
