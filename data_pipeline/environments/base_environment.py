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
    ) -> List[List[Candidate]]:
        """
        This creates two sets of `how_many` candidates that
        are intended to be used as start/end respectively. Take the cartesian product of these
        two sets and you'll have a bunch of valid problems.

        :param how_many int: How many candidates to generate in each candidate set (the result
                             is guaranteed to match this number or the function will run forever)
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[List[Candidate]]: A list of candidate sets, where each has `how_many`
                                      candidates on the table.
        """
        assert (
            self.generated
        ), "Must run generate the environment before requesting additional candidates"
        return self._gen_additional_candidate_sets(how_many, selfcc)

    def gen_neutral_candidates(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[Candidate]:
        """
        Generate a set of collision free neutral poses and corresponding configurations
        (represented as Candidate object)

        :param how_many int: How many neutral poses to generate
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[Candidate]: A list of neutral poses
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
    ) -> List[List[Candidate]]:
        """
        This creates two sets of `how_many` candidates that
        are intended to be used as start/end respectively. Take the cartesian product of these
        two sets and you'll have a bunch of valid problems.

        :param how_many int: How many candidates to generate in each candidate set (the result
                             is guaranteed to match this number or the function will run forever)
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[List[Candidate]]: A list of candidate sets, where each has `how_many`
                                      candidates on the table.
        """
        pass

    @abstractmethod
    def _gen_neutral_candidates(
        self, how_many: int, selfcc: FrankaSelfCollisionChecker
    ) -> List[Candidate]:
        """
        Generate a set of collision free neutral poses and corresponding configurations
        (represented as Candidate object)

        :param how_many int: How many neutral poses to generate
        :param selfcc FrankaSelfCollisionChecker: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[Candidate]: A list of neutral poses
        """
        pass
