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

import numpy as np

from robofin.robots import FrankaRobot, FrankaGripper, FrankaRealRobot
from robofin.bullet import Bullet
import time
import pickle
from pathlib import Path
import itertools
from robofin.collision import FrankaSelfCollisionChecker
from geometrout.transform import SE3, SO3
from termcolor import colored
import logging
from mpinets.types import Obstacles

from geometrout.primitive import Cuboid, Sphere, Cylinder
from typing import Sequence, Union, List, Tuple, Any, Dict, Optional
from mpinets.types import Obstacles, Trajectory
from mpinets.third_party.sparc import sparc

try:
    import lula
except ImportError:
    logging.info("Lula not available. Falling back to bullet only")
    USE_LULA = False
else:
    USE_LULA = True


def percent_true(arr: Sequence) -> float:
    """
    Returns the percent true of a boolean sequence or the percent nonzero of a numerical sequence

    :param arr Sequence: The input sequence
    :rtype float: The percent
    """
    return 100 * np.count_nonzero(arr) / len(arr)


class Evaluator:
    """
    This class can be used to evaluate a whole set of environments and data
    """

    def __init__(self, fabric_urdf_path: str = None, gui: bool = False):
        """
        Initializes the evaluator class

        :param gui bool: Whether to visualize trajectories (and showing visually whether they are failures)
        """
        self.sim = Bullet(gui=False)
        self.selfcc = FrankaSelfCollisionChecker()
        self.hd_sim_robot = self.sim.load_robot(
            FrankaRobot, hd=True, default_prismatic_value=0.025
        )
        self.ld_sim_robot = self.sim.load_robot(
            FrankaRobot, hd=False, default_prismatic_value=0.025
        )

        self.self_collision_sim = Bullet(gui=False)
        self.self_collision_robot = self.self_collision_sim.load_robot(
            FrankaRobot, hd=True, default_prismatic_value=0.025
        )
        self.groups = {}
        self.current_group = None
        self.current_group_key = None
        if fabric_urdf_path is None:
            self.fabric_urdf_path = "/isaac-sim/exts/omni.isaac.motion_generation/motion_policy_configs/franka/lula_franka_gen.urdf"
        else:
            self.fabric_urdf_path = fabric_urdf_path
        if gui:
            self.gui_sim = Bullet(gui=True)
            self.gui_robot = self.gui_sim.load_robot(
                FrankaRobot, hd=True, default_prismatic_value=0.025
            )
            self.gui_gripper = self.gui_sim.load_robot(FrankaGripper)
        else:
            self.gui_sim = None
            self.gui_robot = None
            self.gui_gripper = None

    def run_visuals(
        self,
        trajectory: Trajectory,
        dt: float,
        target: SE3,
        obstacles: Obstacles,
        success: bool,
    ):
        """
        Visualizes a trajectory and changes the color based on whether its a success or failure

        :param trajectory Trajectory: The trajectory to visualize
        :param dt float: The approximate timestep to use when visualizing
        :param target SE3: The target pose (in right gripper frame)
        :param obstacles Obstacles: The obstacles to visualize
        :param success bool: Whether the trajectory was a success
        """
        self.gui_sim.clear_all_obstacles()
        if success:
            obstacle_color = [181 / 255, 132 / 255, 1, 1]
        else:
            obstacle_color = [1, 0.51, 0, 1]
        collision_ids = self.gui_sim.load_primitives(
            # obstacles, [16 / 255, 6 / 255, 54 / 255, 1]
            obstacles,
            obstacle_color,
        )
        self.gui_gripper.marionette(target)
        self.gui_sim.load_cuboid(
            Cuboid(
                center=target.xyz, dims=[0.01, 0.01, 0.01], quaternion=target.so3.wxyz
            ),
            [16 / 255, 6 / 255, 54 / 255, 1],
        )
        for i, q in enumerate(trajectory):
            self.gui_robot.marionette(q)
            collision_points = self.gui_robot.get_collision_points(
                collision_ids, check_self=False
            )
            if len(collision_points):
                self.gui_sim.load_primitives(
                    [Sphere(center=c, radius=0.02) for c in collision_points],
                    [1, 0, 0, 0.5],
                )
                time.sleep(0.25)
            time.sleep(dt)

    def create_new_group(self, key: str):
        """
        Creates a new metric group (for a new setting, for example)

        :param key str: The key for this metric group
        """
        self.groups[key] = {}
        self.current_group_key = key
        self.current_group = self.groups[key]

    if USE_LULA:

        def _get_fabric(self, obstacles: Obstacles) -> lula.Fabric:
            urdf_path = self.fabric_urdf_path
            assert Path(urdf_path).exists()
            fabric_robot_description_path = str(
                Path(__file__).resolve().parent.parent
                / "config"
                / "franka_robot_description.yaml"
            )
            assert Path(fabric_robot_description_path).exists()
            fabric_config_path = str(
                Path(__file__).resolve().parent.parent
                / "config"
                / "franka_fabric_config.yaml"
            )
            assert Path(fabric_config_path).exists()

            # Load robot description.
            robot_description = lula.load_robot(
                fabric_robot_description_path, urdf_path
            )
            fabric_state = lula.create_fabric_state()

            world = lula.create_world()
            for o in obstacles:
                if o.is_zero_volume():
                    continue
                if isinstance(o, Cuboid):
                    box_obstacle_pose = lula.Pose3(o.pose.matrix)
                    box = lula.create_obstacle(lula.Obstacle.Type.CUBE)
                    box.set_attribute(
                        lula.Obstacle.Attribute.SIDE_LENGTHS, np.asarray(o.dims)
                    )
                    world.add_obstacle(box, box_obstacle_pose)
                elif isinstance(o, Cylinder):
                    cylinder_obstacle_pose = lula.Pose3(o.pose.matrix)
                    cylinder = lula.create_obstacle(lula.Obstacle.Type.CYLINDER)
                    cylinder.set_attribute(lula.Obstacle.Attribute.RADIUS, o.radius)
                    cylinder.set_attribute(lula.Obstacle.Attribute.HEIGHT, o.height)
                    world.add_obstacle(cylinder, cylinder_obstacle_pose)

            world_view = world.add_world_view()

            # Create RMPflow configuration.
            fabric_config = lula.create_fabric_config(
                fabric_config_path,
                robot_description,
                "right_gripper",
                world_view,
            )
            # Create RMPflow policy
            fabric = lula.create_fabric(fabric_config)
            return fabric

        def in_collision(self, trajectory: Trajectory, obstacles: Obstacles) -> bool:
            """
            Checks whether the trajectory is in collision according to all including
            collision checkers (using AND between different methods)

            :param trajectory Trajectory: The trajectory
            :param obstacles Obstacles: Obstacles to check
            :rtype bool: Whether there is a collision
            """
            fabric = self._get_fabric(obstacles)
            for i, q in enumerate(trajectory):
                self.hd_sim_robot.marionette(q)
                self.ld_sim_robot.marionette(q)
                if (
                    self.sim.in_collision(self.ld_sim_robot, check_self=True)
                    and self.sim.in_collision(self.hd_sim_robot, check_self=True)
                    and fabric.in_collision_with_obstacle(q.astype(np.double))
                ):
                    return True
            return False

    else:

        def in_collision(self, trajectory: Trajectory, obstacles: Obstacles) -> bool:
            """
            Checks whether the trajectory is in collision according to all including
            collision checkers (using AND between different methods)

            :param trajectory Trajectory: The trajectory
            :param obstacles Obstacles: Obstacles to check
            :rtype bool: Whether there is a collision
            """
            for i, q in enumerate(trajectory):
                self.hd_sim_robot.marionette(q)
                self.ld_sim_robot.marionette(q)
                if self.sim.in_collision(
                    self.ld_sim_robot, check_self=True
                ) and self.sim.in_collision(self.hd_sim_robot, check_self=True):
                    return True
            return False

    def has_self_collision(self, trajectory: Trajectory) -> bool:
        """
        Checks whether there is a self collision (using OR between different methods)

        :param trajectory Trajectory: The trajectory
        :rtype bool: Whether there is a self collision
        """
        for i, q in enumerate(trajectory):
            self.self_collision_robot.marionette(q)
            if self.self_collision_sim.in_collision(
                self.self_collision_robot, check_self=True
            ) or self.selfcc.has_self_collision(q):
                return True
        return False

    def in_collision(self, trajectory: Trajectory, obstacles: Obstacles) -> bool:
        """
        Checks whether the trajectory is in collision according to all including
        collision checkers (using AND between different methods)

        :param trajectory Trajectory: The trajectory
        :param obstacles Obstacles: Obstacles to check
        :rtype bool: Whether there is a collision
        """
        if USE_LULA:
            fabric = self._get_fabric(obstacles)
        for i, q in enumerate(trajectory):
            self.hd_sim_robot.marionette(q)
            self.ld_sim_robot.marionette(q)
            if self.sim.in_collision(
                self.ld_sim_robot, check_self=True
            ) and self.sim.in_collision(self.hd_sim_robot, check_self=True):
                if not USE_LULA:
                    return True
                if fabric.in_collision_with_obstacle(q.astype(np.double)):
                    return True
        return False

    def get_collision_depths(
        self, trajectory: Trajectory, obstacles: Obstacles
    ) -> List[float]:
        """
        Get all the collision depths for a trajectory (sometimes can report strange
        values due to inconsistent Bullet collision checking)

        :param trajectory Trajectory: The trajectory
        :param obstacles Obstacles: The obstacles
        :rtype List[float]: A list of all collision depths
        """
        all_depths = []
        for i, q in enumerate(trajectory):
            self.hd_sim_robot.marionette(q)
            depths = self.hd_sim_robot.get_collision_depths(self.sim.obstacle_ids)
            all_depths.extend(depths)
        return all_depths

    @staticmethod
    def violates_joint_limits(trajectory: Trajectory) -> bool:
        """
        Checks whether any configuration in the trajectory violates joint limits

        :param trajectory Trajectory: The trajectory
        :rtype bool: Whether there is a joint limit violation
        """
        for i, q in enumerate(trajectory):
            if not FrankaRobot.within_limits(q):
                return True
        return False

    def has_physical_violation(
        self, trajectory: Trajectory, obstacles: Obstacles
    ) -> bool:
        """
        Checks whether there is any physical violation (collision, self collision, joint limit violation)

        :param trajectory Trajectory: The trajectory
        :param obstacles Obstacles: The obstacles in the scene
        :rtype bool: Whether there is at least one physical violation
        """
        return (
            self.in_collision(trajectory, obstacles)
            or self.violates_joint_limits(trajectory)
            or self.has_self_collision(trajectory)
        )

    @staticmethod
    def check_final_position(final_pose: SE3, target: SE3) -> float:
        """
        Gets the number of centimeters between the final pose and target

        :param final_pose SE3: The final pose of the trajectory
        :param target SE3: The target pose
        :rtype float: The distance in centimeters
        """
        return 100 * np.linalg.norm(final_pose._xyz - target._xyz)

    @staticmethod
    def check_final_orientation(final_orientation: SO3, target: SO3) -> float:
        """
        Gets the number of degrees between the final orientation and the target orientation

        :param final_orientation SO3: The final orientation
        :param target SO3: The final target orientation
        :rtype float: The rotational distance in degrees
        """
        return np.abs(
            (final_orientation._quat * target._quat.conjugate).radians * 180 / np.pi
        )

    @staticmethod
    def check_final_region(
        final_pose: SE3,
        target_volume: Union[Cuboid, Cylinder, Sphere],
        negative_volumes: Obstacles,
    ) -> bool:
        """
        Checks that the final pose is in the correct region if one is specified. For example,
        when reaching inside a drawer, it is not sufficient to be close to the final target as
        a condition of success. Instead, the final pose might be in the correct drawer
        (and likewise not in the incorrect drawer)

        :param final_pose SE3: The final pose of the trajectory
        :param target_volume Union[Cuboid, Cylinder, Sphere]: The target volume to be inside
        :param negative_volumes Obstacles: Volumes to be outside of
        :rtype bool: Whether the check is successful and the target is in the right region
                     and outside the wrong regions
        """
        return target_volume.sdf(final_pose._xyz) <= 0 and np.all(
            [v.sdf(final_pose._xyz) > 0 for v in negative_volumes]
        )

    @staticmethod
    def calculate_smoothness(trajectory: Trajectory, dt: float) -> Tuple[float, float]:
        """
        Calculate trajectory smoothness using SPARC

        :param trajectory Trajectory: The trajectory
        :param dt float: The timestep in between consecutive steps of the trajectory
        :rtype Tuple[float, float]: The SPARC in configuration space and end effector space
        """
        configs = np.asarray(trajectory)
        assert configs.ndim == 2 and configs.shape[1] == 7
        config_movement = np.linalg.norm(np.diff(configs, 1, axis=0) / dt, axis=1)
        assert len(config_movement) == len(configs) - 1
        config_sparc, _, _ = sparc(config_movement, 1.0 / dt)

        eff_positions = np.asarray(
            [FrankaRobot.fk(q, eff_frame="right_gripper")._xyz for q in trajectory]
        )
        assert eff_positions.ndim == 2 and eff_positions.shape[1] == 3
        eff_movement = np.linalg.norm(np.diff(eff_positions, 1, axis=0) / dt, axis=1)
        assert len(eff_movement) == len(eff_positions) - 1
        eff_sparc, _, _ = sparc(eff_movement, 1.0 / dt)

        return config_sparc, eff_sparc

    def calculate_eff_path_lengths(self, trajectory: Trajectory) -> Tuple[float, float]:
        """
        Calculate the end effector path lengths (position and orientation).
        Orientation is in degrees.

        :param trajectory Trajectory: The trajectory
        :rtype Tuple[float, float]: The path lengths (position, orientation)
        """
        eff_poses = [FrankaRobot.fk(q, eff_frame="right_gripper") for q in trajectory]

        eff_positions = np.asarray([pose._xyz for pose in eff_poses])
        assert eff_positions.ndim == 2 and eff_positions.shape[1] == 3
        position_step_lengths = np.linalg.norm(
            np.diff(eff_positions, 1, axis=0), axis=1
        )
        eff_position_path_length = sum(position_step_lengths)

        eff_quaternions = [pose.so3._quat for pose in eff_poses]
        eff_orientation_path_length = 0
        for qi, qj in zip(eff_quaternions[:-1], eff_quaternions[1:]):
            eff_orientation_path_length += np.abs(
                np.degrees((qj * qi.conjugate).radians)
            )
        return eff_position_path_length, eff_orientation_path_length

    def evaluate_trajectory(
        self,
        trajectory: Trajectory,
        dt: float,
        target: SE3,
        obstacles: Obstacles,
        target_volume: Union[Cuboid, Cylinder, Sphere],
        target_negative_volumes: Obstacles,
        time: float,
        skip_metrics: bool = False,
    ):
        """
        Evaluates a single trajectory and stores the metrics in the current group.
        Will visualize and print relevant info if `self.gui` is `True`

        :param trajectory Trajectory: The trajectory
        :param dt float: The time step for the trajectory
        :param target SE3: The target pose
        :param obstacles Obstacles: The obstacles in the scene
        :param target_volume Union[Cuboid, Cylinder, Sphere]: The target volume for the trajectory
        :param target_negative_volumes Obstacles: Volumes that the target should definitely be outside
        :param time float: The time taken to calculate the trajectory
        :param skip_metrics bool: Whether to skip the path metrics (for example if it's a feasibility planner that failed)
        """

        def add_metric(key, value):
            self.current_group[key] = self.current_group.get(key, []) + [value]

        if skip_metrics:
            add_metric("success", False)
            add_metric("time", np.inf)
            add_metric("skips", True)
            return
        self.sim.clear_all_obstacles()
        self.sim.load_primitives(obstacles)

        in_collision = self.in_collision(trajectory, obstacles)
        if in_collision:
            collision_depths = (
                self.get_collision_depths(trajectory, obstacles) if in_collision else []
            )
        else:
            collision_depths = []
        add_metric("collision_depths", collision_depths)
        add_metric("collision", in_collision)
        add_metric("joint_limit_violation", self.violates_joint_limits(trajectory))
        add_metric("self_collision", self.has_self_collision(trajectory))

        physical_violation = self.has_physical_violation(trajectory, obstacles)
        add_metric("physical_violations", physical_violation)

        final_pose = FrankaRobot.fk(trajectory[-1], eff_frame="right_gripper")

        position_error = self.check_final_position(final_pose, target)
        add_metric("position_error", position_error)

        orientation_error = self.check_final_orientation(final_pose.so3, target.so3)
        add_metric("orientation_error", orientation_error)

        config_smoothness, eff_smoothness = self.calculate_smoothness(trajectory, dt)
        add_metric("config_smoothness", config_smoothness)
        add_metric("eff_smoothness", eff_smoothness)

        (
            eff_position_path_length,
            eff_orientation_path_length,
        ) = self.calculate_eff_path_lengths(trajectory)
        add_metric("eff_position_path_length", eff_position_path_length)
        add_metric("eff_orientation_path_length", eff_orientation_path_length)

        # Sometimes the target is inside a negative volume. This is obviously a bad negative volume
        corrected_negative_volumes = [
            v for v in target_negative_volumes if v.sdf(target._xyz) > 0
        ]
        correct_final_region = self.check_final_region(
            final_pose, target_volume, corrected_negative_volumes
        )

        success = (
            position_error < 1
            and correct_final_region
            and orientation_error < 15
            and not physical_violation
        )

        add_metric("success", success)
        add_metric("time", time)
        add_metric("num_steps", len(trajectory))

        def msg(metric, is_error):
            return colored(metric, "red") if is_error else colored(metric, "green")

        if self.gui_sim is not None and len(collision_depths) > 0:
            print("Position Error:", msg(position_error, position_error > 1))
            print("Orientation Error:", msg(orientation_error, orientation_error > 15))
            print("Config SPARC:", msg(config_smoothness, config_smoothness > -1.6))
            print("End Eff SPARC:", msg(eff_smoothness, eff_smoothness > -1.6))
            print(f"End Eff Position Path Length: {eff_position_path_length}")
            print(f"End Eff Orientation Path Length: {eff_orientation_path_length}")
            print("Physical violation:", msg(physical_violation, physical_violation))
            print(
                "Collisions:",
                msg(
                    self.in_collision(trajectory, obstacles),
                    self.in_collision(trajectory, obstacles),
                ),
            )
            if len(collision_depths) > 0:
                print(f"Mean Collision Depths: {100 * np.max(collision_depths)}")
                print(f"Median Collision Depths: {100 * np.median(collision_depths)}")
            print(
                "Joint Limit Violation:",
                msg(
                    self.violates_joint_limits(trajectory),
                    self.violates_joint_limits(trajectory),
                ),
            )
            print(
                "Self collision:",
                msg(
                    self.has_self_collision(trajectory),
                    self.has_self_collision(trajectory),
                ),
            )
            print(
                "Correct Region:", msg(correct_final_region, not correct_final_region)
            )
            self.run_visuals(trajectory, dt, target, obstacles, success)

    @staticmethod
    def metrics(group: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculates the metrics for a specific group

        :param group Dict[str, Any]: The group of results
        :rtype Dict[str, float]: The metrics
        """
        success = percent_true(group["success"])
        one_cm = percent_true(np.array(group["position_error"]) < 1)
        five_cm = percent_true(np.array(group["position_error"]) < 5)
        fifteen_deg = percent_true(np.array(group["orientation_error"]) < 15)
        thirty_deg = percent_true(np.array(group["orientation_error"]) < 30)
        turn_around = percent_true(np.array(group["orientation_error"]) > 165)

        physical = percent_true(group["physical_violations"])
        config_smoothness = np.mean(group["config_smoothness"])
        eff_smoothness = np.mean(group["eff_smoothness"])
        all_eff_position_path_lengths = np.asarray(group["eff_position_path_length"])
        all_eff_orientation_path_lengths = np.asarray(
            group["eff_orientation_path_length"]
        )
        all_times = np.asarray(group["time"])

        is_smooth = percent_true(
            np.logical_and(
                np.asarray(group["config_smoothness"]) < -1.6,
                np.asarray(group["eff_smoothness"]) < -1.6,
            )
        )

        # Only use filtered successes for eff path length because we only
        # save the metric when it's an unskipped trajectory
        # TODO maybe clean this up so that it's not so if/else-y and either
        # all metrics are saved or only the ones that aren't skipped save
        skips = []
        if "skips" in group:
            # Needed when there are skips
            successes = np.asarray(group["success"])
            unskipped_successes = successes[~np.isinf(all_times)]
            skips = group["skips"]
        else:
            unskipped_successes = group["success"]

        success_position_path_lengths = all_eff_position_path_lengths[
            unskipped_successes
        ]
        success_orientation_path_lengths = all_eff_orientation_path_lengths[
            unskipped_successes
        ]
        eff_position_path_length = (
            np.mean(success_position_path_lengths),
            np.std(success_position_path_lengths),
        )
        eff_orientation_path_length = (
            np.mean(success_orientation_path_lengths),
            np.std(success_orientation_path_lengths),
        )
        success_times = all_times[group["success"]]
        time = (
            np.mean(success_times),
            np.std(success_times),
        )

        collision = percent_true(group["collision"])
        joint_limit = percent_true(group["joint_limit_violation"])
        self_collision = percent_true(group["self_collision"])
        depths = np.array(
            list(itertools.chain.from_iterable(group["collision_depths"]))
        )
        all_num_steps = np.asarray(group["num_steps"])
        success_num_steps = all_num_steps[unskipped_successes]
        step_time = (
            np.mean(success_times / success_num_steps),
            np.std(success_times / success_num_steps),
        )
        return {
            "success": success,
            "total": len(group["success"]),
            "skips": len(skips),
            "time": time,
            "step time": step_time,
            "env collision": collision,
            "self collision": self_collision,
            "joint violation": joint_limit,
            "physical violations": physical,
            "average collision depth": 100 * np.mean(depths),
            "median collision depth": 100 * np.median(depths),
            "1 cm": one_cm,
            "5 cm": five_cm,
            "15 deg": fifteen_deg,
            "30 deg": thirty_deg,
            "165 deg": turn_around,
            "is smooth": is_smooth,
            "average config sparc": config_smoothness,
            "average eff sparc": eff_smoothness,
            "eff position path length": eff_position_path_length,
            "eff orientation path length": eff_orientation_path_length,
        }

    @staticmethod
    def print_metrics(group: Dict[str, Any]):
        """
        Prints the metrics in an easy to read format

        :param group Dict[str, float]: The group of results
        """
        metrics = Evaluator.metrics(group)
        print(f"Total problems: {metrics['total']}")
        print(f"# Skips (Hard Failures): {metrics['skips']}")
        print(f"% Success: {metrics['success']:4.2f}")
        print(f"% Within 1cm: {metrics['1 cm']:4.2f}")
        print(f"% Within 5cm: {metrics['5 cm']:4.2f}")
        print(f"% Within 15deg: {metrics['15 deg']:4.2f}")
        print(f"% Within 30deg: {metrics['30 deg']:4.2f}")
        print(f"% Within 15deg of 180: {metrics['165 deg']:4.2f}")
        print(f"% With Environment Collision: {metrics['env collision']:4.2f}")
        print(f"% With Self Collision: {metrics['self collision']:4.2f}")
        print(f"% With Joint Limit Violations: {metrics['joint violation']:4.2f}")
        print(f"% With Self Collision: {metrics['self collision']:4.2f}")
        print(f"Average Collision Depth (cm): {metrics['average collision depth']}")
        print(f"Median Collision Depth (cm): {metrics['median collision depth']}")
        print(f"% With Physical Violations: {metrics['physical violations']:4.2f}")
        print(f"Average Config SPARC: {metrics['average config sparc']:4.2f}")
        print(f"Average End Eff SPARC: {metrics['average eff sparc']:4.2f}")
        print(f"% Smooth: {metrics['is smooth']:4.2f}")
        print(
            "Average End Eff Position Path Length:"
            f" {metrics['eff position path length'][0]:4.2f}"
            f" ± {metrics['eff position path length'][1]:4.2f}"
        )
        print(
            "Average End Eff Orientation Path Length:"
            f" {metrics['eff orientation path length'][0]:4.2f}"
            f" ± {metrics['eff orientation path length'][1]:4.2f}"
        )
        print(f"Average Time: {metrics['time'][0]:4.2f} ± {metrics['time'][1]:4.2f}")
        print(
            "Average Time Per Step (Not Always Valuable):"
            f" {metrics['step time'][0]:4.6f}"
            f" ± {metrics['step time'][1]:4.6f}"
        )

    def save_group(self, directory: str, test_name: str, key: Optional[str] = None):
        """
        Save the results of a single group

        :param directory str: The directory in which to save the results
        :param test_name str: The name of this specific test
        :param key Optional[str]: The group key to use. If not specified will use the current group
        """
        if key is None:
            group = self.current_group
        else:
            group = self.groups[key]
        save_path = Path(directory) / f"{test_name}_{self.current_group_key}.pkl"
        print(f"Saving group metrics to {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(group, f)

    def save(self, directory: str, test_name: str):
        """
        Save all the groups

        :param directory str: The directory name in which to save results
        :param test_name str: The test name (used as the file name)
        """
        save_path = Path(directory) / f"{test_name}_metrics.pkl"
        print(f"Metrics will save to {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(self.groups, f)

    def print_group_metrics(self, key: Optional[str] = None):
        """
        Prints out the metrics for a specific group

        :param key Optional[str]: The group key (if none specified, will use current group)
        """
        if key is not None:
            self.current_group = self.groups[key]
            self.current_group_key = key
        return self.print_metrics(self.current_group)

    def print_overall_metrics(self):
        """
        Prints the metrics for the aggregated results over all groups
        """
        supergroup = {}
        keys = set()
        for group in self.groups.values():
            for key in group.keys():
                keys.add(key)

        for key in keys:
            metrics = []
            for group in self.groups.values():
                metrics.extend(group.get(key, []))
            supergroup[key] = metrics
        return self.print_metrics(supergroup)
