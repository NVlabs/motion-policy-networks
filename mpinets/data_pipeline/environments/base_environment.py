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

from abc import (
    ABC,
    abstractmethod,
)
from typing import List, Union, List, Any
from robofin.collision import FrankaSelfCollisionChecker
from geometrout.primitive import Cuboid, Cylinder
from dataclasses import dataclass, field
from geometrout.transform import SE3
import numpy as np
from mpinets.types import Obstacles


def radius_sample(center: float, radius: float):
    """
    Helper function to draw a uniform sample with a fixed radius around a center

    :param center float: The center of the distribution
    :param radius float: The radius of the distribution
    """
    return np.random.uniform(center - radius, center + radius)


@dataclass
class Candidate:
    """
    Represents a configuration and the corresponding end-effector pose
    (in the right_gripper frame).
    """

    pose: SE3
    config: np.ndarray
    negative_volumes: Obstacles


@dataclass
class TaskOrientedCandidate(Candidate):
    """
    Represents a configuration and the corresponding end-effector pose
    (in the right_gripper frame) for a task oriented pose.
    """

    pass


@dataclass
class NeutralCandidate(Candidate):
    """
    Represents a configuration and the corresponding end-effector pose
    (in the right_gripper frame) for a neutral pose.
    """

    pass


class Environment(ABC):
    def __init__(self):
        self.generated = False
        self.demo_candidates = []
        pass

    @property
    @abstractmethod
    def obstacles(self) -> List[Union[Cuboid, Cylinder]]:
        """
        Returns all obstacles in the scene.
        :rtype List[Union[Cuboid, Cylinder]]: The list of obstacles in the scene
        """
        pass

    @property
    @abstractmethod
    def cuboids(self) -> List[Cuboid]:
        """
        Returns just the cuboids in the scene
        :rtype List[Cuboid]: The list of cuboids in the scene
        """
        pass

    @property
    @abstractmethod
    def cylinders(self) -> List[Cylinder]:
        """
        Returns just the cylinders in the scene
        :rtype List[Cylinder]: The list of cylinders in the scene
        """
        pass

    def gen(self, selfcc: FrankaSelfCollisionChecker, **kwargs: Any) -> bool:
        """
        Generates an environment and a pair of start/end candidates

        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype bool: Whether the environment was successfully generated
        """
        self.generated = self._gen(selfcc, **kwargs)
        if self.generated:
            assert len(self.demo_candidates) == 2
            cand1, cand2 = self.demo_candidates
            assert cand1 is not None and cand2 is not None
        return self.generated

    def gen_additional_candidate_sets(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[List[TaskOrientedCandidate]]:
        """
        This creates two sets of `how_many` candidates that
        are intended to be used as start/end respectively. Take the cartesian product of these
        two sets and you'll have a bunch of valid problems.

        :param how_many int: How many candidates to generate in each candidate set (the result
                             is guaranteed to match this number or the function will run forever)
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[List[TaskOrientedCandidate]]: A list of candidate sets, where each has `how_many`
                                      candidates on the table.
        """
        assert (
            self.generated
        ), "Must run generate the environment before requesting additional candidates"
        return self._gen_additional_candidate_sets(how_many, selfcc)

    def gen_neutral_candidates(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[NeutralCandidate]:
        """
        Generate a set of collision free neutral poses and corresponding configurations
        (represented as Candidate object)

        :param how_many int: How many neutral poses to generate
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[NeutralCandidate]: A list of neutral poses
        """
        assert (
            self.generated
        ), "Must run generate the environment before requesting additional candidates"
        return self._gen_neutral_candidates(how_many, selfcc)

    @abstractmethod
    def _gen(self, selfcc: FrankaSelfCollisionChecker) -> bool:
        """
        The internal implementation of the gen function.

        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype bool: Whether the environment was successfully generated
        """
        pass

    @abstractmethod
    def _gen_additional_candidate_sets(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[List[TaskOrientedCandidate]]:
        """
        This creates two sets of `how_many` candidates that
        are intended to be used as start/end respectively. Take the cartesian product of these
        two sets and you'll have a bunch of valid problems.

        :param how_many int: How many candidates to generate in each candidate set (the result
                             is guaranteed to match this number or the function will run forever)
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[List[TaskOrientedCandidate]]: A list of candidate sets, where each has `how_many`
                                      candidates on the table.
        """
        pass

    @abstractmethod
    def _gen_neutral_candidates(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[NeutralCandidate]:
        """
        Generate a set of collision free neutral poses and corresponding configurations
        (represented as NeutralCandidate object)

        :param how_many int: How many neutral poses to generate
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[NeutralCandidate]: A list of neutral poses
        """
        pass
