#!/usr/bin/env python3

import torch
from mpinets.model import MotionPolicyNetwork
from robofin.robots import FrankaRealRobot
from robofin.pointcloud.torch import FrankaSampler
import numpy as np
from mpinets.utils import normalize_franka_joints, unnormalize_franka_joints
from mpinets_msgs.msg import PlanningProblem
from sensor_msgs.msg import PointCloud2, PointField
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
import time
import trimesh.transformations as tra
from functools import partial
from geometrout.transform import SE3
import argparse
from typing import List, Tuple, Any

import rospy

NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 75


class Planner:
    @torch.no_grad()
    def __init__(self, mdl_file: str):
        """
        Initializes and loads the model from the checkpoint

        :param mdl_file str: The path to the model checkpoint to be loaded
        """
        """
        """
        self.mdl = MotionPolicyNetwork.load_from_checkpoint(mdl_file).cuda().eval()
        self.fk_sampler = FrankaSampler("cuda:0")

    @torch.no_grad()
    def target_point_cloud(self, pose: SE3) -> torch.Tensor:
        """
        Samples target points on the gripper

        :param pose SE3: pose of gripper in world frame
        :rtype torch.Tensor: A point cloud sampled from the gripper's mesh
        """
        target_points = self.fk_sampler.sample_end_effector(
            torch.as_tensor(pose.matrix).float().cuda().unsqueeze(0),
            num_points=NUM_TARGET_POINTS,
        )
        return target_points

    @torch.no_grad()
    def plan(
        self, q0: np.ndarray, target_pose: SE3, obstacle_pc: np.ndarray
    ) -> Tuple[bool, List[List[float]]]:
        """
        Creates a trajectory rollout toward the target. Will give up after MAX_ROLLOUT_LENGTH
        prediction steps

        :param q0 np.ndarray: A 7D array (dim 7,) representing the starting config
        :param target_pose SE3: A target pose in the `right_gripper` frame
        :param obstacle_pc np.ndarray: All the obstacle points fed to the network. These should
                            be constructed by filtering out outlier points and randomly
                            downsampling to be of length NUM_OBSTACLE_POINTS
        :rtype List[List[float]]: A trajectory as a list of lists (each has 7D). Formatted
                                  as a list to be more friendly to the ROS publisher
        """
        assert obstacle_pc.shape == (NUM_OBSTACLE_POINTS, 3), (
            "You must downsample obstacle PC before passing to planner. "
            "While you're at it, filter the outliers out as well"
        )
        obstacle_points = torch.as_tensor(obstacle_pc).cuda()
        target_points = self.target_point_cloud(target_pose).squeeze()
        assert np.all(
            FrankaRealRobot.JOINT_LIMITS[:, 0] <= q0
        ), "Configuration is outside of feasible limits"
        assert np.all(
            q0 <= FrankaRealRobot.JOINT_LIMITS[:, 1]
        ), "Configuration is outside of feasible limits"
        q = torch.as_tensor(q0).cuda().unsqueeze(0).float()
        robot_points = self.fk_sampler.sample(q, NUM_ROBOT_POINTS)
        point_cloud = torch.cat(
            (
                torch.zeros(NUM_ROBOT_POINTS, 4),
                torch.ones(NUM_OBSTACLE_POINTS, 4),
                2 * torch.ones(NUM_TARGET_POINTS, 4),
            ),
            dim=0,
        ).cuda()
        point_cloud[:NUM_ROBOT_POINTS, :3] = robot_points.float()
        point_cloud[
            NUM_ROBOT_POINTS : NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS, :3
        ] = obstacle_points.float()
        point_cloud[
            NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS :, :3
        ] = target_points.float()
        point_cloud = point_cloud.unsqueeze(0)

        trajectory = [q]
        q_norm = normalize_franka_joints(q)
        success = False
        for _ in range(MAX_ROLLOUT_LENGTH):
            step_start = time.time()
            q_norm = torch.clamp(q_norm + self.mdl(point_cloud, q_norm), min=-1, max=1)
            qt = unnormalize_franka_joints(q_norm).type_as(q)
            assert isinstance(qt, torch.Tensor)
            trajectory.append(qt)
            eff_pose = FrankaRealRobot.fk(
                qt.squeeze().detach().cpu().numpy(), eff_frame="right_gripper"
            )
            # [TUNE] This is where the 'success' is defined.
            # Feel free to change this.
            if (
                np.linalg.norm(eff_pose._xyz - target_pose._xyz) < 0.01
                and np.abs(
                    np.degrees(
                        (eff_pose.so3._quat * target_pose.so3._quat.conjugate).radians
                    )
                )
                < 15
            ):
                success = True
                break
            robot_points = self.fk_sampler.sample(qt, NUM_ROBOT_POINTS)
            point_cloud[:, :NUM_ROBOT_POINTS, :3] = robot_points
        return success, [q.squeeze().cpu().numpy().tolist() for q in trajectory]


class PlanningNode:
    def __init__(self):
        """
        Initializes the subscribers, loads the data from file, and loads the model.
        """
        rospy.init_node("mpinets_planning_node")
        time.sleep(1)

        self.planner = None
        self.base_frame = "panda_link0"
        self.planning_problem_subscriber = rospy.Subscriber(
            "/mpinets/planning_problem",
            PlanningProblem,
            self.plan_callback,
            queue_size=1,
        )
        self.full_point_cloud_publisher = rospy.Publisher(
            "/mpinets/full_point_cloud", PointCloud2, queue_size=2
        )
        self.plan_publisher = rospy.Publisher(
            "/mpinets/plan", JointTrajectory, queue_size=1
        )
        rospy.loginfo("Loading data")
        self.load_point_cloud_data(
            rospy.get_param("/mpinets_planning_node/point_cloud_path")
        )
        rospy.loginfo("Data loaded")
        rospy.loginfo("Loading model")
        self.planner = Planner(rospy.get_param("/mpinets_planning_node/mdl_path"))
        rospy.loginfo("Model loaded")
        rospy.loginfo("System ready")

    @staticmethod
    def clean_point_cloud(
        xyz: np.ndarray, rgba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Some points are outside of the feasible range and create artifacts for
        the network. This filters out those points and then downsamples to the right size
        for the network

        :param xyz np.ndarray: The geometry information for the point cloud (dim N x 3)
        :param rgba np.ndarray: The color information for the point cloud (dim N x 4)
        :rtype Tuple[np.ndarray, np.ndarray]: Returns a tuple of the cleaned and
                                              downsized geometry information and color
                                              information
        """
        task_tabletop_mask = np.logical_and.reduce(
            (
                xyz[:, 0] > 0.25,
                xyz[:, 0] < 1.35,
                xyz[:, 1] > -0.3,
                xyz[:, 1] < 1.6,
                xyz[:, 2] > -0.05,
                xyz[:, 2] < 0.35,
            )
        )

        mount_table_mask = np.logical_and.reduce(
            (
                xyz[:, 0] > -0.35,
                xyz[:, 0] < 0.30,
                xyz[:, 1] > -0.5,
                xyz[:, 1] < 0.5,
                xyz[:, 2] > -0.05,
                xyz[:, 2] < 0.05,
            )
        )
        workspace_mask = np.logical_or(task_tabletop_mask, mount_table_mask)
        xyz = xyz[workspace_mask]
        rgba = rgba[workspace_mask]
        random_mask = np.random.choice(
            len(xyz), size=NUM_OBSTACLE_POINTS, replace=False
        )
        return xyz[random_mask], rgba[random_mask]

    def load_point_cloud_data(self, path: str):
        """
        Loads scene from a point cloud file, transforms into the
        'panda_link0' frame, stores it to the class, and starts a publishing
        loop to show it

        :param path str: The path to the point cloud file
        """

        # Load the file
        observation_data = np.load(
            path,
            allow_pickle=True,
        ).item()

        # Transform it into the "world frame," i.e. `panda_link0`
        full_pc = tra.transform_points(
            observation_data["pc"], observation_data["camera_pose"]
        )

        # Remove the robot points
        no_robot_mask = (
            observation_data["label_map"]["robot"] != observation_data["pc_label"]
        )
        scene_pc = full_pc[no_robot_mask]

        # Scale the color values to be within [0-1] and add alpha channel
        scene_colors = observation_data["pc_color"][no_robot_mask] / 255.0
        scene_colors = np.concatenate(
            (scene_colors, np.ones((len(scene_colors), 1))), axis=1
        )
        assert scene_colors.shape[1] == 4
        rospy.Timer(
            rospy.Duration(1.0),
            partial(self.publish_point_cloud_data, scene_pc, scene_colors),
        )
        self.full_scene_pc = scene_pc
        self.full_scene_colors = scene_colors

    def publish_point_cloud_data(self, points: np.ndarray, colors: np.ndarray, _: Any):
        """
        Publishes the point cloud so that it can be visualized in Rviz

        :param points np.ndarray: The 3D locations of the point cloud (dimension N x 3)
        :param colors np.ndarray: The color values of each point (dimension N x 4)
        :param _ Any: This is a parameter necessary to run this within a rospy timing
                      loop and is unused.
        """
        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize
        assert points.shape[1] == 3
        assert colors.shape[1] == 4
        colors[:, -1] = 0.5
        data = np.concatenate((points, colors), axis=1).astype(dtype)  # .tobytes()
        data = data.tobytes()
        fields = [
            PointField(name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate("xyzrgba")
        ]
        header = Header(frame_id="panda_link0", stamp=rospy.Time.now())
        msg = PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 7),
            row_step=(itemsize * 7 * points.shape[0]),
            data=data,
        )
        self.full_point_cloud_publisher.publish(msg)

    def plan_callback(self, msg: PlanningProblem):
        """
        Receives the planning problem from the interaction tool and calls the planner
        Afterward, it publishes the solution, whether or not we consider it a success

        :param msg PlanningProblem: A message describing the planning problem
        """
        q0 = np.asarray(msg.q0.position)
        target = SE3(
            xyz=[
                msg.target.transform.translation.x,
                msg.target.transform.translation.y,
                msg.target.transform.translation.z,
            ],
            quat=[
                msg.target.transform.rotation.w,
                msg.target.transform.rotation.x,
                msg.target.transform.rotation.y,
                msg.target.transform.rotation.z,
            ],
        )
        scene_pc, scene_colors = self.clean_point_cloud(
            self.full_scene_pc, self.full_scene_colors
        )
        if self.planner is None:
            rospy.logwarn("Model is not yet loaded and planner cannot yet be called")
            return
        rospy.loginfo(f"Attempting to plan")
        success, plan = self.planner.plan(q0, target, scene_pc)
        rospy.loginfo(f"Planning succeeded: {success}")
        joint_trajectory = JointTrajectory()
        joint_trajectory.header.stamp = rospy.Time.now()
        joint_trajectory.header.frame_id = "panda_link0"
        joint_trajectory.joint_names = msg.joint_names
        for ii, q in enumerate(plan):
            point = JointTrajectoryPoint(
                time_from_start=rospy.Duration.from_sec(0.12 * ii)
            )
            for qi in q:
                point.positions.append(qi)
            joint_trajectory.points.append(point)
        rospy.loginfo("Planning solution published")
        self.plan_publisher.publish(joint_trajectory)


if __name__ == "__main__":
    PlanningNode()
    rospy.spin()
