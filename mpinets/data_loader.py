from pathlib import Path
from typing import Optional, List, Union, Dict
import enum
import os

from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import numpy as np
import torch
import pytorch_lightning as pl
from pyquaternion import Quaternion
from geometrout.primitive import Cuboid, Cylinder
from robofin.pointcloud.torch import FrankaSampler

from robofin.robots import FrankaRealRobot
from mpinets.geometry import construct_mixed_point_cloud
from mpinets import utils


class DatasetType(enum.Enum):
    """
    A simple enum class to indicate whether a dataloader is for training, validating, or testing
    """

    TRAIN = 0
    VAL = 1
    TEST = 2


class PointCloudBase(Dataset):
    """
    This base class should never be used directly, but it handles the filesystem
    management and the basic indexing. When using these dataloaders, the directory
    holding the data should look like so:
        directory/
          train/
             train.hdf5
          val/
             val.hdf5
          test/
             test.hdf5
    Note that only the relevant subdirectory is required, i.e. when creating a
    dataset for training, this class will not check for (and will not use) the val/
    and test/ subdirectories.
    """

    def __init__(
        self,
        directory: Path,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        dataset_type: DatasetType,
        random_scale: float,
    ):
        """
        :param directory Path: The path to the root of the data directory
        :param num_robot_points int: The number of points to sample from the robot
        :param num_obstacle_points int: The number of points to sample from the obstacles
        :param num_target_points int: The number of points to sample from the target
                                      robot end effector
        :param dataset_type DatasetType: What type of dataset this is
        :param random_scale float: The standard deviation of the random normal
                                   noise to apply to the joints during training.
                                   This is only used for train datasets.
        """
        self._init_directory(directory, dataset_type)
        self.trajectory_key = trajectory_key
        self.train = dataset_type == DatasetType.TRAIN
        with h5py.File(str(self._database), "r") as f:
            self._num_trajectories = f[self.trajectory_key].shape[0]
            self.expert_length = f[self.trajectory_key].shape[1]

        self.num_obstacle_points = num_obstacle_points
        self.num_robot_points = num_robot_points
        self.num_target_points = num_target_points
        self.random_scale = random_scale
        self.fk_sampler = FrankaSampler("cpu", use_cache=True)

    def _init_directory(self, directory: Path, dataset_type: DatasetType):
        """
        Sets the path for the internal data structure based on the dataset type

        :param directory Path: The path to the root of the data directory
        :param dataset_type DatasetType: What type of dataset this is
        :raises Exception: Raises an exception when the dataset type is unsupported
        """
        self.type = dataset_type
        if dataset_type == DatasetType.TRAIN:
            directory = directory / "train"
        elif dataset_type == DatasetType.VAL:
            directory = directory / "val"
        elif dataset_type == DatasetType.TEST:
            directory = directory / "test"
        else:
            raise Exception(f"Invalid dataset type: {dataset_type}")

        databases = list(directory.glob("**/*.hdf5"))
        assert len(databases) == 1
        self._database = databases[0]

    @property
    def num_trajectories(self):
        """
        Returns the total number of trajectories in the dataset
        """
        return self._num_trajectories

    @staticmethod
    def normalize(configuration_tensor: torch.Tensor):
        """
        Normalizes the joints between -1 and 1 according the the joint limits

        :param configuration_tensor torch.Tensor: The input tensor. Has dim [7]
        """
        return utils.normalize_franka_joints(configuration_tensor)

    def get_inputs(self, trajectory_idx: int, timestep: int) -> Dict[str, torch.Tensor]:
        """
        Loads all the relevant data and puts it in a dictionary. This includes
        normalizing all configurations and constructing the pointcloud.
        If a training dataset, applies some randomness to joints (before
        sampling the pointcloud).

        :param trajectory_idx int: The index of the trajectory in the hdf5 file
        :param timestep int: The timestep within that trajectory
        :rtype Dict[str, torch.Tensor]: The data used aggregated by the dataloader
                                        and used for training
        """
        item = {}
        with h5py.File(str(self._database), "r") as f:
            target_pose = FrankaRealRobot.fk(
                f[self.trajectory_key][trajectory_idx, -1, :]
            )
            target_points = self.fk_sampler.sample_end_effector(
                torch.as_tensor(target_pose.matrix).float(),
                num_points=self.num_target_points,
            )
            item["target_position"] = torch.as_tensor(target_pose.xyz).float()

            config = f[self.trajectory_key][trajectory_idx, timestep, :]
            config_tensor = torch.as_tensor(config).float()

            if self.train:
                # Add slight random noise to the joints
                randomized = (
                    self.random_scale * torch.randn(config_tensor.shape) + config_tensor
                )
                # Ensure that after adding random noise, the joint angles are still within the joint limits
                limits = torch.as_tensor(FrankaRealRobot.JOINT_LIMITS).float()

                # Clamp to joint limits
                randomized = torch.minimum(
                    torch.maximum(randomized, limits[:, 0]), limits[:, 1]
                )
                item["configuration"] = self.normalize(randomized)
                robot_points = self.fk_sampler.sample(randomized, self.num_robot_points)
            else:
                item["configuration"] = self.normalize(config_tensor)
                robot_points = self.fk_sampler.sample(
                    config_tensor, self.num_robot_points
                )

            cuboid_dims = f["cuboid_dims"][trajectory_idx, ...]
            if cuboid_dims.ndim == 1:
                cuboid_dims = np.expand_dims(cuboid_dims, axis=0)

            cuboid_centers = f["cuboid_centers"][trajectory_idx, ...]
            if cuboid_centers.ndim == 1:
                cuboid_centers = np.expand_dims(cuboid_centers, axis=0)

            cuboid_quats = f["cuboid_quaternions"][trajectory_idx, ...]
            if cuboid_quats.ndim == 1:
                cuboid_quats = np.expand_dims(cuboid_quats, axis=0)
            # Entries without a shape are stored with an invalid quaternion of all zeros
            # This will cause NaNs later in the pipeline. It's best to set these to unit
            # quaternions.
            # To find invalid shapes, we just look for a dimension with size 0
            cuboid_quats[np.all(np.isclose(cuboid_quats, 0), axis=1), 0] = 1

            # Leaving in the zero volume cuboids to conform to a standard
            # Pytorch array size. These have to be filtered out later
            item["cuboid_dims"] = torch.as_tensor(cuboid_dims)
            item["cuboid_centers"] = torch.as_tensor(cuboid_centers)
            item["cuboid_quats"] = torch.as_tensor(cuboid_quats)

            if "cylinder_radii" not in f.keys():
                # Create a dummy cylinder if cylinders aren't in the hdf5 file
                cylinder_radii = np.array([[0.0]])
                cylinder_heights = np.array([[0.0]])
                cylinder_centers = np.array([[0.0, 0.0, 0.0]])
                cylinder_quats = np.array([[1.0, 0.0, 0.0, 0.0]])
            else:
                cylinder_radii = f["cylinder_radii"][trajectory_idx, ...]
                if cylinder_radii.ndim == 1:
                    cylinder_radii = np.expand_dims(cylinder_radii, axis=0)
                cylinder_heights = f["cylinder_heights"][trajectory_idx, ...]
                if cylinder_heights.ndim == 1:
                    cylinder_heights = np.expand_dims(cylinder_heights, axis=0)
                cylinder_centers = f["cylinder_centers"][trajectory_idx, ...]
                if cylinder_centers.ndim == 1:
                    cylinder_centers = np.expand_dims(cylinder_centers, axis=0)
                cylinder_quats = f["cylinder_quaternions"][trajectory_idx, ...]
                if cylinder_quats.ndim == 1:
                    cylinder_quats = np.expand_dims(cylinder_quats, axis=0)
                # Ditto to the comment above about fixing ill-formed quaternions
                cylinder_quats[np.all(np.isclose(cylinder_quats, 0), axis=1), 0] = 1

            item["cylinder_radii"] = torch.as_tensor(cylinder_radii)
            item["cylinder_heights"] = torch.as_tensor(cylinder_heights)
            item["cylinder_centers"] = torch.as_tensor(cylinder_centers)
            item["cylinder_quats"] = torch.as_tensor(cylinder_quats)

            cuboids = [
                Cuboid(c, d, q)
                for c, d, q in zip(
                    list(cuboid_centers), list(cuboid_dims), list(cuboid_quats)
                )
            ]

            # Filter out the cuboids with zero volume
            cuboids = [c for c in cuboids if not c.is_zero_volume()]

            cylinders = [
                Cylinder(c, r, h, q)
                for c, r, h, q in zip(
                    list(cylinder_centers),
                    list(cylinder_radii.squeeze(1)),
                    list(cylinder_heights.squeeze(1)),
                    list(cylinder_quats),
                )
            ]
            cylinders = [c for c in cylinders if not c.is_zero_volume()]

            obstacle_points = construct_mixed_point_cloud(
                cuboids + cylinders, self.num_obstacle_points
            )
            item["xyz"] = torch.cat(
                (
                    torch.zeros(self.num_robot_points, 4),
                    torch.ones(self.num_obstacle_points, 4),
                    2 * torch.ones(self.num_target_points, 4),
                ),
                dim=0,
            )
            item["xyz"][: self.num_robot_points, :3] = robot_points.float()
            item["xyz"][
                self.num_robot_points : self.num_robot_points
                + self.num_obstacle_points,
                :3,
            ] = torch.as_tensor(obstacle_points[:, :3]).float()
            item["xyz"][
                self.num_robot_points + self.num_obstacle_points :,
                :3,
            ] = target_points.float()

        return item


class PointCloudTrajectoryDataset(PointCloudBase):
    """
    This dataset is used exclusively for validating. Each element in the dataset
    represents a trajectory start and scene. There is no supervision because
    this is used to produce an entire rollout and check for success. When doing
    validation, we care more about success than we care about matching the
    expert's behavior (which is a key difference from training).
    """

    def __init__(
        self,
        directory: Path,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        dataset_type: DatasetType,
    ):
        """
        :param directory Path: The path to the root of the data directory
        :param num_robot_points int: The number of points to sample from the robot
        :param num_obstacle_points int: The number of points to sample from the obstacles
        :param num_target_points int: The number of points to sample from the target
                                      robot end effector
        :param dataset_type DatasetType: What type of dataset this is
        """
        assert (
            dataset_type != DatasetType.TRAIN
        ), "This dataset is not meant for training"
        super().__init__(
            directory,
            trajectory_key,
            num_robot_points,
            num_obstacle_points,
            num_target_points,
            dataset_type,
            random_scale=0.0,
        )

    def __len__(self):
        """
        Necessary for Pytorch. For this dataset, the length is the total number
        of problems
        """
        return self.num_trajectories

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Required by Pytorch. Queries for data at a particular index. Note that
        in this dataset, the index always corresponds to the trajectory index.

        :param idx int: The index
        :rtype Dict[str, torch.Tensor]: Returns a dictionary that can be assembled
            by the data loader before using in training.
        """
        trajectory_idx, timestep = idx, 0
        item = self.get_inputs(trajectory_idx, timestep)

        return item


class PointCloudInstanceDataset(PointCloudBase):
    """
    This is the dataset used primarily for training. Each element in the dataset
    represents the robot and scene at a particular time $t$. Likewise, the
    supervision is the robot's configuration at q_{t+1}.
    """

    def __init__(
        self,
        directory: Path,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        dataset_type: DatasetType,
        random_scale: float,
    ):
        """
        :param directory Path: The path to the root of the data directory
        :param num_robot_points int: The number of points to sample from the robot
        :param num_obstacle_points int: The number of points to sample from the obstacles
        :param num_target_points int: The number of points to sample from the target
                                      robot end effector
        :param dataset_type DatasetType: What type of dataset this is
        :param random_scale float: The standard deviation of the random normal
                                   noise to apply to the joints during training.
                                   This is only used for train datasets.
        """
        super().__init__(
            directory,
            trajectory_key,
            num_robot_points,
            num_obstacle_points,
            num_target_points,
            dataset_type,
            random_scale,
        )

    def __len__(self):
        """
        Returns the total number of start configurations in the dataset (i.e.
        the length of the trajectories times the number of trajectories)

        """
        return self.num_trajectories * self.expert_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a training datapoint representing a single configuration in a
        single scene with the configuration at the next timestep as supervision

        :param idx int: Index represents the timestep within the trajectory
        :rtype Dict[str, torch.Tensor]: The data used for training
        """
        trajectory_idx, timestep = divmod(idx, self.expert_length)
        if timestep >= self.expert_length:
            timestep = self.expert_length - 1
        item = self.get_inputs(trajectory_idx, timestep)

        # Re-use the last point in the trajectory at the end
        supervision_timestep = np.clip(
            timestep + 1,
            0,
            self.expert_length - 1,
        )

        with h5py.File(str(self._database), "r") as f:
            item["supervision"] = self.normalize(
                torch.as_tensor(
                    f[self.trajectory_key][trajectory_idx, supervision_timestep, :]
                )
            ).float()

        return item


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        random_scale: float,
        batch_size: int,
    ):
        """
        :param data_dir str: The directory with the data. Directory structure should
                             be as defined in `PointCloudBase`
        :param trajectory_key str: The key in the hdf5 dataset that contains the expert trajectories
        :param num_robot_points int: The number of points to sample from the robot
        :param num_obstacle_points int: The number of points to sample from the obstacles
        :param num_target_points int: The number of points to sample from the target
                                      robot end effector
        :param random_scale float: The standard deviation of the random normal
                                   noise to apply to the joints during training.
        :param batch_size int: The batch size
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.trajectory_key = trajectory_key
        self.batch_size = batch_size
        self.num_robot_points = num_robot_points
        self.num_obstacle_points = num_obstacle_points
        self.num_target_points = num_target_points
        self.num_workers = os.cpu_count()
        self.random_scale = random_scale

    def setup(self, stage: Optional[str] = None):
        """
        A Pytorch Lightning method that is called per-device in when doing
        distributed training.

        :param stage Optional[str]: Indicates whether we are in the training
                                    procedure or if we are doing ad-hoc testing
        """
        if stage == "fit" or stage is None:
            self.data_train = PointCloudInstanceDataset(
                self.data_dir,
                self.trajectory_key,
                self.num_robot_points,
                self.num_obstacle_points,
                self.num_target_points,
                dataset_type=DatasetType.TRAIN,
                random_scale=self.random_scale,
            )
            self.data_val = PointCloudTrajectoryDataset(
                self.data_dir,
                self.trajectory_key,
                self.num_robot_points,
                self.num_obstacle_points,
                self.num_target_points,
                dataset_type=DatasetType.VAL,
            )
        if stage == "test" or stage is None:
            self.data_test = PointCloudInstanceDataset(
                self.data_dir,
                self.trajectory_key,
                self.num_robot_points,
                self.num_obstacle_points,
                self.num_target_points,
                dataset_type=DatasetType.TEST,
                random_scale=self.random_scale,
            )

    def train_dataloader(self) -> DataLoader:
        """
        A Pytorch lightning method to get the dataloader for training

        :rtype DataLoader: The training dataloader
        """
        return DataLoader(
            self.data_train,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        A Pytorch lightning method to get the dataloader for validation

        :rtype DataLoader: The validation dataloader
        """
        return DataLoader(
            self.data_val,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        A Pytorch lightning method to get the dataloader for testing

        :rtype DataLoader: The dataloader for testing
        """
        return DataLoader(
            self.data_test,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
