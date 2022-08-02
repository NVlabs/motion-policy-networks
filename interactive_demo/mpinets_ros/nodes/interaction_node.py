#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import (
    Marker,
    InteractiveMarkerFeedback,
    InteractiveMarker,
    InteractiveMarkerControl,
)
from sensor_msgs.msg import PointCloud2, PointField
from mpinets_msgs.msg import PlanningProblem
import ctypes
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
import struct
import numpy as np
import time
import tf2_ros
import tf_conversions
from copy import deepcopy

# Maybe these can be removed if we have a different way of sending commands
import sys
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState


# The neutral configuration at which to start the node
NEUTRAL_CONFIG = np.array(
    [
        -0.01779206,
        -0.76012354,
        0.01978261,
        -2.34205014,
        0.02984053,
        1.54119353,
        0.75344866,
        0.025,
        0.025,
    ]
)

# A neutral starting target (matches the end effector of the neutral start)
NEUTRAL_TARGET_XYZ = [0.30649957127333377, 0.007287351995245575, 0.4866376674460814]
NEUTRAL_TARGET_XYZW = [
    -0.014241942613215233,
    0.9996573431202568,
    0.008466023255748396,
    -0.02026548461261383,
]

# The joint names. These would ideally be read from the URDF, but they are hard-coded
# here out of laziness since this is meant to be a demo
JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
]


class MPiNetsInterface:
    def __init__(self):
        """
        Initialize the system state, the interactive components, and the subscribers/publishers
        """
        rospy.init_node("mpinets_interface")
        self.server = InteractiveMarkerServer("mpinets_controls", "")
        self.br = tf2_ros.TransformBroadcaster()
        # rospy.Timer(rospy.Duration.from_sec(0.01), self.frame_callback)
        # Not sure why this is here, but ok

        self.make_execute_button_marker([1.0, -1.0, 0.1], 0.2)
        self.make_plan_button_marker([0.7, -1.0, 0.1], 0.2)
        self.make_reset_button_marker([0.4, -1.0, 0.1], 0.2)

        self.target_pose = None
        self.visualize_plan = False
        self.current_plan = []
        self.target_xyz = NEUTRAL_TARGET_XYZ
        self.target_xyzw = NEUTRAL_TARGET_XYZW
        self.current_joint_state = NEUTRAL_CONFIG
        self.make_target_marker(
            self.target_xyz,
            self.target_xyzw,
        )
        self.server.applyChanges()
        self.planning_problem_publisher = rospy.Publisher(
            "/mpinets/planning_problem", PlanningProblem, queue_size=1
        )
        self.joint_states_publisher = rospy.Publisher(
            "/mpinets/joint_states",
            JointState,
            queue_size=1,
        )

        self.planned_joint_states_publisher = rospy.Publisher(
            "/mpinets/planned_joint_states",
            JointState,
            queue_size=1,
        )

        self.planning_problem_subscriber = rospy.Subscriber(
            "/mpinets/plan",
            JointTrajectory,
            self.planning_callback,
            queue_size=5,
        )
        time.sleep(1)
        self.reset_franka()

    def reset_franka(self):
        """
        Resets the robot to the neutral pose and resets the current plan
        """
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "panda_link0"
        msg.position = NEUTRAL_CONFIG
        msg.name = JOINT_NAMES
        self.visualize_plan = False
        self.joint_states_publisher.publish(msg)
        self.planned_joint_states_publisher.publish(msg)
        self.current_joint_state = NEUTRAL_CONFIG
        self.current_plan = []

    @staticmethod
    def make_box(side_length, color):
        """
        Makes a colored box that can be viewed in Rviz (will be used as buttons)
        """
        marker = Marker()
        marker.type = Marker.CUBE

        marker.scale.x = side_length
        marker.scale.y = side_length
        marker.scale.z = side_length
        (
            marker.color.r,
            marker.color.g,
            marker.color.b,
            marker.color.a,
        ) = color
        return marker

    @staticmethod
    def make_gripper(msg):
        """
        Creates a floating gripper that can be viewed in Rviz (will be used as the target)
        """
        marker = Marker()
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = "package://mpinets_ros/meshes/half_open_gripper.stl"

        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        return marker

    def make_reset_button_marker(self, xyz, side_length):
        """
        Creates a red cube that resets the system when you click on it

        :param xyz List[float]: The center of the button cube
        :param side_length float: The side length for the cube
        """
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "panda_link0"
        (
            int_marker.pose.position.x,
            int_marker.pose.position.y,
            int_marker.pose.position.z,
        ) = xyz
        int_marker.scale = 0.5

        int_marker.name = "reset_button"
        int_marker.description = "Reset"

        control = InteractiveMarkerControl()

        control.interaction_mode = InteractiveMarkerControl.BUTTON
        control.name = "reset_button_control"

        marker = self.make_box(side_length, [204.0 / 255, 50.0 / 255, 50.0 / 255, 1.0])
        control.markers.append(marker)
        control.always_visible = True
        int_marker.controls.append(control)
        # from IPython import embed
        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.reset_button_callback)

    def make_plan_button_marker(self, xyz, side_length):
        """
        Create a yellow cube that calls the planner and visualizes the result when you click on it

        :param xyz List[float]: The center of the button cube
        :param side_length float: The side length for the cube
        """
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "panda_link0"
        (
            int_marker.pose.position.x,
            int_marker.pose.position.y,
            int_marker.pose.position.z,
        ) = xyz
        int_marker.scale = 0.5

        int_marker.name = "plan_button"
        int_marker.description = "Plan"

        control = InteractiveMarkerControl()

        control.interaction_mode = InteractiveMarkerControl.BUTTON
        control.name = "plan_button_control"

        marker = self.make_box(side_length, [231.0 / 255, 180.0 / 255, 22.0 / 255, 1.0])
        control.markers.append(marker)
        control.always_visible = True
        int_marker.controls.append(control)
        # from IPython import embed
        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.plan_button_callback)

    def make_execute_button_marker(self, xyz, side_length):
        """
        Create a green cube button that executes on the robot when you click on it

        :param xyz List[float]: The center of the button cube
        :param side_length float: The side length for the cube
        """
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "panda_link0"
        (
            int_marker.pose.position.x,
            int_marker.pose.position.y,
            int_marker.pose.position.z,
        ) = xyz
        int_marker.scale = 0.5

        int_marker.name = "execute_button"
        int_marker.description = "Execute"

        control = InteractiveMarkerControl()

        control.interaction_mode = InteractiveMarkerControl.BUTTON
        control.name = "execute_button_control"

        marker = self.make_box(side_length, [45.0 / 255, 201.0 / 255, 55.0 / 255, 1.0])
        control.markers.append(marker)
        control.always_visible = True
        int_marker.controls.append(control)
        # from IPython import embed
        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.execute_button_callback)

    def make_gripper_control(self, msg):
        """
        Creates the gripper marker for the target

        :param msg InteractiveMarker: The interactive marker for the target
        :rtype InteractiveMarkerControl: The gripper control handle for the target marker
        """
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(self.make_gripper(msg))
        msg.controls.append(control)
        return msg.controls[-1]

    def make_target_marker(self, xyz, xyzw):
        """
        Create the target interactive marker

        :param xyz List[float]: The starting position for the target marker
        :param xyzw List[float]: The starting orientation for the target marker
        """
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "panda_link0"
        (
            int_marker.pose.position.x,
            int_marker.pose.position.y,
            int_marker.pose.position.z,
        ) = xyz
        (
            int_marker.pose.orientation.x,
            int_marker.pose.orientation.y,
            int_marker.pose.orientation.z,
            int_marker.pose.orientation.w,
        ) = xyzw
        int_marker.scale = 0.4

        int_marker.name = "target"
        int_marker.description = "Trajectory Target"

        self.make_gripper_control(int_marker)
        int_marker.controls[0].interaction_mode = InteractiveMarkerControl.NONE

        control = InteractiveMarkerControl()
        xyzw = np.array([1.0, 0.0, 0.0, 1.0])
        xyzw = xyzw / np.linalg.norm(xyzw)
        (
            control.orientation.x,
            control.orientation.y,
            control.orientation.z,
            control.orientation.w,
        ) = xyzw
        control.name = "rotate_x"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(deepcopy(control))
        control.name = "move_x"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(deepcopy(control))

        xyzw = np.array([0.0, 1.0, 0.0, 1.0])
        xyzw = xyzw / np.linalg.norm(xyzw)
        (
            control.orientation.x,
            control.orientation.y,
            control.orientation.z,
            control.orientation.w,
        ) = xyzw
        control.name = "rotate_z"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(deepcopy(control))
        control.name = "move_z"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(deepcopy(control))

        xyzw = np.array([0.0, 0.0, 1.0, 1.0])
        xyzw = xyzw / np.linalg.norm(xyzw)
        (
            control.orientation.x,
            control.orientation.y,
            control.orientation.z,
            control.orientation.w,
        ) = xyzw
        control.name = "rotate_y"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(deepcopy(control))
        control.name = "move_y"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(deepcopy(control))

        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.target_feedback)

    def reset_button_callback(self, feedback):
        """
        A callback that's called after clicking on the reset button, resets the system

        :param feedback InteractiveMarkerFeedback: The feedback from interacting with the
                                                   reset button
        """
        print(type(feedback))
        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            rospy.loginfo("Resetting robot to neutral pose")
            self.reset_franka()
        self.server.applyChanges()

    def planning_callback(self, msg):
        """
        Displays the planned trajectory as a ghost

        :param msg JointTrajectory: The trajectory coming back from the planning node
        """
        self.current_plan = [
            list(point.positions) + NEUTRAL_CONFIG[7:].tolist() for point in msg.points
        ]
        while self.visualize_plan:
            for q in self.current_plan:
                joint_msg = JointState()
                joint_msg.header.stamp = rospy.Time.now()
                joint_msg.header.frame_id = "panda_link0"
                joint_msg.position = q
                joint_msg.name = JOINT_NAMES
                # Checking one more time in case there's a race condition
                if self.visualize_plan:
                    self.planned_joint_states_publisher.publish(joint_msg)
                rospy.sleep(0.12)

    def plan_button_callback(self, feedback):
        """
        This is called whenever the plan button is clicked. It publishes a planning problem
        that can then be solved by the planning node. In a more production-ready system,
        this would use a Service instead of just publishing and subscribing, but in this
        implementation, it uses a publisher to reduce boilerplate.

        :param feedback InteractiveMarkerFeedback: The feedback from interacting with the
                                                   plan button
        """
        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            msg = PlanningProblem()
            msg.header.stamp = rospy.Time.now()
            msg.joint_names = JOINT_NAMES[:7]
            msg.target = TransformStamped()
            msg.target.header.frame_id = "panda_link0"
            msg.target.child_frame_id = "target_frame"
            (
                msg.target.transform.translation.x,
                msg.target.transform.translation.y,
                msg.target.transform.translation.z,
            ) = self.target_xyz
            (
                msg.target.transform.rotation.x,
                msg.target.transform.rotation.y,
                msg.target.transform.rotation.z,
                msg.target.transform.rotation.w,
            ) = self.target_xyzw
            msg.q0 = JointState(position=self.current_joint_state[:7])
            self.visualize_plan = True
            self.planning_problem_publisher.publish(msg)

        self.server.applyChanges()

    def execute_button_callback(self, feedback):
        """
        This is called whenever the execute button is clicked. It will move the "real"
        robot according to the currently calculated plan. It will also reset the plan and
        current "real" joint configuration to the end of the plan.

        :param feedback InteractiveMarkerFeedback: The feedback from interacting with the
                                                   execute button
        """
        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            if len(self.current_plan) == 0:
                rospy.logwarn("There is no current plan. Plan before executing")
                return
            self.visualize_plan = False
            self.current_joint_state = self.current_plan[-1]
            for q in self.current_plan:
                joint_msg = JointState()
                joint_msg.header.stamp = rospy.Time.now()
                joint_msg.header.frame_id = "panda_link0"
                joint_msg.position = q
                joint_msg.name = JOINT_NAMES
                self.joint_states_publisher.publish(joint_msg)
                self.planned_joint_states_publisher.publish(joint_msg)
                rospy.sleep(0.12)
            self.current_plan = []

        self.server.applyChanges()

    def target_feedback(self, feedback):
        """
        This is called whenever the user interacts with the target marker. This is used to
        set the target pose.

        :param feedback InteractiveMarkerFeedback: The feedback from interacting with the
                                                   execute button
        """
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            self.target_xyz = (
                feedback.pose.position.x,
                feedback.pose.position.y,
                feedback.pose.position.z,
            )
            self.target_xyzw = (
                feedback.pose.orientation.x,
                feedback.pose.orientation.y,
                feedback.pose.orientation.z,
                feedback.pose.orientation.w,
            )


if __name__ == "__main__":
    env = MPiNetsInterface()
    rospy.spin()
