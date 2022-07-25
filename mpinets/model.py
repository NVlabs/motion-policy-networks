import torch
from torch import nn
from robofin.pointcloud.torch import FrankaSampler, FrankaCollisionSampler
import pytorch_lightning as pl
from pointnet2_ops.pointnet2_modules import PointnetSAModule

from mpinets import loss
from mpinets.utils import unnormalize_franka_joints
from mpinets.geometry import TorchCuboids, TorchCylinders
from typing import List, Tuple, Sequence, Dict, Callable


class MotionPolicyNetwork(pl.LightningModule):
    """
    The architecture laid out here is the default architecture laid out in the
    Motion Policy Networks paper (Fishman, et. al, 2022).
    """

    def __init__(self):
        """
        Constructs the model
        """
        super().__init__()
        self.point_cloud_encoder = MPiNetsPointNet()
        self.feature_encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2048 + 64, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 7),
        )

    def configure_optimizers(self):
        """
        A standard method in PyTorch lightning to set the optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def forward(self, xyz: torch.Tensor, q: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Passes data through the network to produce an output

        :param xyz torch.Tensor: Tensor representing the point cloud. Should
                                      have dimensions of [B x N x 4] where B is the batch
                                      size, N is the number of points and 4 is because there
                                      are three geometric dimensions and a segmentation mask
        :param q torch.Tensor: The current robot configuration normalized to be between
                                    -1 and 1, according to each joint's range of motion
        :rtype torch.Tensor: The displacement to be applied to the current configuration to get
                     the position at the next step (still in normalized space)
        """
        pc_encoding = self.point_cloud_encoder(xyz)
        feature_encoding = self.feature_encoder(q)
        x = torch.cat((pc_encoding, feature_encoding), dim=1)
        return self.decoder(x)


class TrainingMotionPolicyNetwork(MotionPolicyNetwork):
    """
    An version of the MotionPolicyNetwork model that has additional attributes
    necessary during training (or using the validation step outside of the
    training process). This class is a valid model, but it's overkill when
    doing real robot inference and, for example, point cloud sampling is
    done by an outside process (such as downsampling point clouds from a point cloud).
    """

    def __init__(
        self,
        num_robot_points: int,
        point_match_loss_weight: float,
        collision_loss_weight: float,
    ):
        """
        Creates the network and assigns additional parameters for training


        :param num_robot_points int: The number of robot points used when resampling
                                     the robot points during rollouts (used in validation)
        :param point_match_loss_weight float: The weight assigned to the behavior
                                              cloning loss.
        :param collision_loss_weight float: The weight assigned to the collision loss
        :rtype Self: An instance of the network
        """
        super().__init__()
        self.num_robot_points = num_robot_points
        self.point_match_loss_weight = point_match_loss_weight
        self.collision_loss_weight = collision_loss_weight
        self.fk_sampler = None
        self.collision_sampler = None
        self.loss_fun = loss.CollisionAndBCLossContainer()

    def rollout(
        self,
        batch: Dict[str, torch.Tensor],
        rollout_length: int,
        sampler: Callable[[torch.Tensor], torch.Tensor],
        unnormalize: bool = False,
    ) -> List[torch.Tensor]:
        """
        Rolls out the policy an arbitrary length by calling it iteratively

        :param batch Dict[str, torch.Tensor]: A data batch coming from the
                                                   data loader--should already be
                                                   on the correct device
        :param rollout_length int: The number of steps to roll out (not including the start)
        :param sampler Callable[[torch.Tensor], torch.Tensor]: A function that takes a batch of robot
                                                                         configurations [B x 7] and returns a batch of
                                                                         point clouds samples on the surface of that robot
        :param unnormalize bool: Whether to return the whole trajectory unnormalized
                                 (i.e. converted back into joint space)
        :rtype List[torch.Tensor]: The entire trajectory batch, i.e. a list of
                                        configuration batches including the starting
                                        configurations where each element in the list
                                        corresponds to a timestep. For example, the
                                        first element of each batch in the list would
                                        be a single trajectory.
        """
        xyz, q = (
            batch["xyz"],
            batch["configuration"],
        )
        # This block is to adapt for the case where we only want to roll out a
        # single trajectory
        if q.ndim == 1:
            xyz = xyz.unsqueeze(0)
            q = q.unsqueeze(0)
        if unnormalize:
            q_unnorm = unnormalize_franka_joints(q)
            assert isinstance(q_unnorm, torch.Tensor)
            trajectory = [q_unnorm]
        else:
            trajectory = [q]

        for i in range(rollout_length):
            q = torch.clamp(q + self(xyz, q), min=-1, max=1)
            q_unnorm = unnormalize_franka_joints(q)
            assert isinstance(q_unnorm, torch.Tensor)
            q_unnorm = q_unnorm.type_as(q)
            if unnormalize:
                trajectory.append(q_unnorm)
            else:
                trajectory.append(q)

            samples = sampler(q_unnorm).type_as(xyz)
            xyz[:, : samples.shape[1], :3] = samples

        return trajectory

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        A function called automatically by Pytorch Lightning during training.
        This function handles the forward pass, the loss calculation, and what to log

        :param batch Dict[str, torch.Tensor]: A data batch coming from the
                                                   data loader--should already be
                                                   on the correct device
        :param batch_idx int: The index of the batch (not used by this function)
        :rtype torch.Tensor: The overall weighted loss (used for backprop)
        """
        xyz, q = (
            batch["xyz"],
            batch["configuration"],
        )
        y_hat = torch.clamp(q + self(xyz, q), min=-1, max=1)
        (
            cuboid_centers,
            cuboid_dims,
            cuboid_quats,
            cylinder_centers,
            cylinder_radii,
            cylinder_heights,
            cylinder_quats,
            supervision,
        ) = (
            batch["cuboid_centers"],
            batch["cuboid_dims"],
            batch["cuboid_quats"],
            batch["cylinder_centers"],
            batch["cylinder_radii"],
            batch["cylinder_heights"],
            batch["cylinder_quats"],
            batch["supervision"],
        )
        collision_loss, point_match_loss = self.loss_fun(
            y_hat,
            cuboid_centers,
            cuboid_dims,
            cuboid_quats,
            cylinder_centers,
            cylinder_radii,
            cylinder_heights,
            cylinder_quats,
            supervision,
        )
        self.log("point_match_loss", point_match_loss)
        self.log("collision_loss", collision_loss)
        val_loss = (
            self.point_match_loss_weight * point_match_loss
            + self.collision_loss_weight * collision_loss
        )
        self.log("val_loss", val_loss)
        return val_loss

    def sample(self, q: torch.Tensor) -> torch.Tensor:
        """
        Samples a point cloud from the surface of all the robot's links

        :param q torch.Tensor: Batched configuration in joint space
        :rtype torch.Tensor: Batched point cloud of size [B, self.num_robot_points, 3]
        """
        assert self.fk_sampler is not None
        return self.fk_sampler.sample(q, self.num_robot_points)

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        This is a Pytorch Lightning function run automatically across devices
        during the validation loop

        :param batch Dict[str, torch.Tensor]: The batch coming from the dataloader
        :param batch_idx int: The index of the batch (not used by this function)
        :rtype torch.Tensor: The loss values which are to be collected into summary stats
        """

        # These are defined here because they need to be set on the correct devices.
        # The easiest way to do this is to do it at call-time
        if self.fk_sampler is None:
            self.fk_sampler = FrankaSampler(self.device, use_cache=True)
        if self.collision_sampler is None:
            self.collision_sampler = FrankaCollisionSampler(
                self.device, with_base_link=False
            )
        rollout = self.rollout(batch, 69, self.sample, unnormalize=True)

        assert self.fk_sampler is not None  # Necessary for mypy to type properly
        eff = self.fk_sampler.end_effector_pose(rollout[-1])
        position_error = torch.linalg.vector_norm(
            eff[:, :3, -1] - batch["target_position"], dim=1
        )
        avg_target_error = torch.mean(position_error)

        cuboids = TorchCuboids(
            batch["cuboid_centers"],
            batch["cuboid_dims"],
            batch["cuboid_quats"],
        )
        cylinders = TorchCylinders(
            batch["cylinder_centers"],
            batch["cylinder_radii"],
            batch["cylinder_heights"],
            batch["cylinder_quats"],
        )

        B = batch["cuboid_centers"].size(0)
        rollout = torch.stack(rollout, dim=1)
        # Here is some Pytorch broadcasting voodoo to calculate whether each
        # rollout has a collision or not (looking to calculate the collision rate)
        assert rollout.shape == (B, 70, 7)
        rollout = rollout.reshape(-1, 7)
        has_collision = torch.zeros(B, dtype=torch.bool, device=self.device)
        collision_spheres = self.collision_sampler.compute_spheres(rollout)
        for radius, spheres in collision_spheres:
            num_spheres = spheres.shape[-2]
            sphere_sequence = spheres.reshape((B, -1, num_spheres, 3))
            sdf_values = torch.minimum(
                cuboids.sdf_sequence(sphere_sequence),
                cylinders.sdf_sequence(sphere_sequence),
            )
            assert sdf_values.shape == (B, 70, num_spheres)
            radius_collisions = torch.any(
                sdf_values.reshape((sdf_values.size(0), -1)) <= radius, dim=-1
            )
            has_collision = torch.logical_or(radius_collisions, has_collision)

        avg_collision_rate = torch.count_nonzero(has_collision) / B
        return {
            "avg_target_error": avg_target_error,
            "avg_collision_rate": avg_collision_rate,
        }

    def validation_step_end(  # type: ignore[override]
        self, batch_parts: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Called by Pytorch Lightning at the end of each validation step to
        aggregate across devices

        :param batch_parts Dict[str, torch.Tensor]: The parts accumulated from all devices
        :rtype Dict[str, torch.Tensor]: The average values across the devices
        """
        return {
            "avg_target_error": torch.mean(batch_parts["avg_target_error"]),
            "avg_collision_rate": torch.mean(batch_parts["avg_collision_rate"]),
        }

    def validation_epoch_end(  # type: ignore[override]
        self, validation_step_outputs: Sequence[Dict[str, torch.Tensor]]
    ):
        """
        Pytorch lightning method that aggregates stats from the validation loop and logs

        :param validation_step_outputs Sequence[Dict[str, torch.Tensor]]: The outputs from each
                                                                      validation step
        """
        avg_target_error = torch.mean(
            torch.stack([x["avg_target_error"] for x in validation_step_outputs])
        )
        self.log("avg_target_error", avg_target_error)

        avg_collision_rate = torch.mean(
            torch.stack([x["avg_collision_rate"] for x in validation_step_outputs])
        )
        self.log("avg_collision_rate", avg_collision_rate)


class MPiNetsPointNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self._build_model()

    def _build_model(self):
        """
        Assembles the model design into a ModuleList
        """
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.05,
                nsample=128,
                mlp=[1, 64, 64, 64],
                bn=False,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.3,
                nsample=128,
                mlp=[64, 128, 128, 256],
                bn=False,
            )
        )
        self.SA_modules.append(PointnetSAModule(mlp=[256, 512, 512, 1024], bn=False))

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.GroupNorm(16, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.GroupNorm(16, 2048),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2048, 2048),
        )

    @staticmethod
    def _break_up_pc(pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Breaks up the point cloud into the xyz coordinates and segmentation mask

        :param pc torch.Tensor: Tensor with shape [B, N, M] where M is larger than 3.
                                The first three dimensions along the last axis will be x, y, z
        :rtype Tuple[torch.Tensor, torch.Tensor]: Two tensors, one with just xyz
            and one with the corresponding features
        """
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous()
        return xyz, features

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass of the network

        :param point_cloud torch.Tensor: Has dimensions (B, N, 4)
                                              B is the batch size
                                              N is the number of points
                                              4 is x, y, z, segmentation_mask
                                              This tensor must be on the GPU (CPU tensors not supported)
        :rtype torch.Tensor: The output from the network
        """
        assert point_cloud.size(2) == 4
        xyz, features = self._break_up_pc(point_cloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))
