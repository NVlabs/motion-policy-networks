from typing import Tuple
from mpinets import utils
from mpinets.geometry import TorchCuboids, TorchCylinders
import torch.nn.functional as F
import torch
from robofin.pointcloud.torch import FrankaSampler


def point_match_loss(input_pc: torch.Tensor, target_pc: torch.Tensor) -> torch.Tensor:
    """
    A combination L1 and L2 loss to penalize large and small deviations between
    two point clouds

    :param input_pc torch.Tensor: Point cloud sampled from the network's output.
                                  Has dim [B, N, 3]
    :param target_pc torch.Tensor: Point cloud sampled from the supervision
                                   Has dim [B, N, 3]
    :rtype torch.Tensor: The single loss value
    """
    return F.mse_loss(input_pc, target_pc, reduction="mean") + F.l1_loss(
        input_pc, target_pc, reduction="mean"
    )


def collision_loss(
    input_pc: torch.Tensor,
    cuboid_centers: torch.Tensor,
    cuboid_dims: torch.Tensor,
    cuboid_quaternions: torch.Tensor,
    cylinder_centers: torch.Tensor,
    cylinder_radii: torch.Tensor,
    cylinder_heights: torch.Tensor,
    cylinder_quaternions: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates the hinge loss, calculating whether the robot (represented as a
    point cloud) is in collision with any obstacles in the scene. Collision
    here actually means within 3cm of the obstacle--this is to provide stronger
    gradient signal to encourage the robot to move out of the way. Also, some of the
    primitives can have zero volume (i.e. a dim is zero for cuboids or radius or height is zero for cylinders).
    If these are zero volume, they will have infinite sdf values (and therefore be ignored by the loss).

    :param input_pc torch.Tensor: Points sampled from the robot's surface after it
                                  is placed at the network's output prediction. Has dim [B, N, 3]
    :param cuboid_centers torch.Tensor: Has dim [B, M1, 3]
    :param cuboid_dims torch.Tensor: Has dim [B, M1, 3]
    :param cuboid_quaternions torch.Tensor: Has dim [B, M1, 4]. Quaternion is formatted as w, x, y, z.
    :param cylinder_centers torch.Tensor: Has dim [B, M2, 3]
    :param cylinder_radii torch.Tensor: Has dim [B, M2, 1]
    :param cylinder_heights torch.Tensor: Has dim [B, M2, 1]
    :param cylinder_quaternions torch.Tensor: Has dim [B, M2, 4]. Quaternion is formatted as w, x, y, z.
    :rtype torch.Tensor: Returns the loss value aggregated over the batch
    """

    cuboids = TorchCuboids(
        cuboid_centers,
        cuboid_dims,
        cuboid_quaternions,
    )
    cylinders = TorchCylinders(
        cylinder_centers,
        cylinder_radii,
        cylinder_heights,
        cylinder_quaternions,
    )
    sdf_values = torch.minimum(cuboids.sdf(input_pc), cylinders.sdf(input_pc))
    return F.hinge_embedding_loss(
        sdf_values,
        -torch.ones_like(sdf_values),
        margin=0.03,
        reduction="mean",
    )


class CollisionAndBCLossContainer:
    """
    A container class to hold the various losses. This is structured as a
    container because that allows it to cache the robot pointcloud sampler
    object. By caching this, we reduce parsing time when processing the URDF
    and allow for a consistent random pointcloud (consistent per-GPU, that is)
    """

    def __init__(
        self,
    ):
        self.fk_sampler = None
        self.num_points = 1024

    def __call__(
        self,
        input_normalized: torch.Tensor,
        cuboid_centers: torch.Tensor,
        cuboid_dims: torch.Tensor,
        cuboid_quaternions: torch.Tensor,
        cylinder_centers: torch.Tensor,
        cylinder_radii: torch.Tensor,
        cylinder_heights: torch.Tensor,
        cylinder_quaternions: torch.Tensor,
        target_normalized: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method calculates both constituent loss function after loading,
        and then caching, a fixed robot point cloud sampler (i.e. the task
        spaces sampled are always the same, as opposed to a random point cloud).
        The fixed point cloud is important for loss calculation so that
        it's possible to take mse between the two pointclouds.

        :param input_normalized torch.Tensor: Has dim [B, 7] and is always between -1 and 1
        :param cuboid_centers torch.Tensor: Has dim [B, M1, 3]
        :param cuboid_dims torch.Tensor: Has dim [B, M1, 3]
        :param cuboid_quaternions torch.Tensor: Has dim [B, M1, 4]. Quaternion is formatted as w, x, y, z.
        :param cylinder_centers torch.Tensor: Has dim [B, M2, 3]
        :param cylinder_radii torch.Tensor: Has dim [B, M2, 1]
        :param cylinder_heights torch.Tensor: Has dim [B, M2, 1]
        :param cylinder_quaternions torch.Tensor: Has dim [B, M2, 4]. Quaternion is formatted as w, x, y, z.
        :param target_normalized torch.Tensor: Has dim [B, 7] and is always between -1 and 1
        :rtype Tuple[torch.Tensor, torch.Tensor]: The two losses aggregated over the batch
        """
        if self.fk_sampler is None:
            self.fk_sampler = FrankaSampler(
                input_normalized.device,
                num_fixed_points=self.num_points,
                use_cache=True,
                with_base_link=False,  # Remove base link because this isn't controllable anyway
            )
        input_pc = self.fk_sampler.sample(
            utils.unnormalize_franka_joints(input_normalized),
        )
        target_pc = self.fk_sampler.sample(
            utils.unnormalize_franka_joints(target_normalized),
        )
        return (
            collision_loss(
                input_pc,
                cuboid_centers,
                cuboid_dims,
                cuboid_quaternions,
                cylinder_centers,
                cylinder_radii,
                cylinder_heights,
                cylinder_quaternions,
            ),
            point_match_loss(input_pc, target_pc),
        )
