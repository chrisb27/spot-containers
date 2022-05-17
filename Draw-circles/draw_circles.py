# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

import argparse
import math
import sys
import time

import numpy as np

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import robot_command_pb2
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.util import seconds_to_duration
from bosdyn.api import (arm_surface_contact_pb2, arm_surface_contact_service_pb2, geometry_pb2,
                        trajectory_pb2)
from bosdyn.client import math_helpers

# User-set params
# duration of the whole move [s]
_SECONDS_FULL = 15
# length of the square the robot walks [m]
_L_ROBOT_SQUARE = 0.5
# length of the square the robot walks [m]
_L_ARM_CIRCLE = 0.4
# shift the circle that the robot draws in z [m]
_VERTICAL_SHIFT = 0

def surface_contact():
    # Position of the hand:
    hand_x_start  = 0.75  # in front of the robot.
    hand_y_start = 0  # centered
    hand_y_end = -0.5  # to the right
    hand_z = 0  # will be ignored since we'll have a force in the Z axis.

    force_z = -0.05  # percentage of maximum press force, negative to press down
    # be careful setting this too large, you can knock the robot over
    percentage_press = geometry_pb2.Vec3(x=0, y=0, z=force_z)

    hand_vec3_start_rt_body = geometry_pb2.Vec3(x=hand_x_start, y=hand_y_start, z=hand_z)
    hand_vec3_end_rt_body = hand_vec3_start_rt_body

    # We want to point the hand straight down the entire time.
    qw = 0.707
    qx = 0
    qy = 0.707
    qz = 0
    body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

    # Build a position trajectory
    body_T_hand1 = geometry_pb2.SE3Pose(position=hand_vec3_start_rt_body,
                                        rotation=body_Q_hand)
    body_T_hand2 = geometry_pb2.SE3Pose(position=hand_vec3_end_rt_body,
                                        rotation=body_Q_hand)

    robot_state = robot_state_client.get_robot_state()
    odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                     ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
    odom_T_hand1 = odom_T_flat_body * math_helpers.SE3Pose.from_obj(body_T_hand1)
    odom_T_hand2 = odom_T_flat_body * math_helpers.SE3Pose.from_obj(body_T_hand2)

    # Trajectory length
    trajectory_time = 5.0  # in seconds
    time_since_reference = seconds_to_duration(trajectory_time)

    traj_point1 = trajectory_pb2.SE3TrajectoryPoint(
        pose=odom_T_hand1.to_proto(), time_since_reference=seconds_to_duration(0))
    traj_point2 = trajectory_pb2.SE3TrajectoryPoint(
        pose=odom_T_hand2.to_proto(), time_since_reference=time_since_reference)

    hand_traj = trajectory_pb2.SE3Trajectory(points=[traj_point1, traj_point2])

    # Open the gripper
    gripper_cmd_packed = RobotCommandBuilder.claw_gripper_open_fraction_command(0)
    gripper_command = gripper_cmd_packed.synchronized_command.gripper_command.claw_gripper_command

    radius = 0.06
    x_ = np.arange(hand_x_start - radius - 1, hand_x_start + radius + 1, dtype=int)
    y_ = np.arange(hand_y_start - radius - 1, hand_y_start + radius + 1, dtype=int)
    _N_POINTS = x_.size


    for ii in range(_N_POINTS + 1):
        cmd = arm_surface_contact_pb2.ArmSurfaceContact.Request(
            pose_trajectory_in_task=hand_traj,
            root_frame_name=ODOM_FRAME_NAME,
            press_force_percentage=percentage_press,
            x_axis=arm_surface_contact_pb2.ArmSurfaceContact.Request.AXIS_MODE_POSITION,
            y_axis=arm_surface_contact_pb2.ArmSurfaceContact.Request.AXIS_MODE_POSITION,
            z_axis=arm_surface_contact_pb2.ArmSurfaceContact.Request.AXIS_MODE_FORCE,
            z_admittance=arm_surface_contact_pb2.ArmSurfaceContact.Request.
                ADMITTANCE_SETTING_LOOSE,
            # Enable the cross term so that if the arm gets stuck in a rut, it will retract
            # upwards slightly, preventing excessive lateral forces.
            xy_to_z_cross_term_admittance=arm_surface_contact_pb2.ArmSurfaceContact.Request.
                ADMITTANCE_SETTING_VERY_STIFF,
            gripper_command=gripper_command)

        # Enable walking
        cmd.is_robot_following_hand = True

        # A bias force (in this case, leaning forward) can help improve stability.
        bias_force_x = -25
        cmd.bias_force_ewrt_body.CopyFrom(geometry_pb2.Vec3(x=x_[ii], y=y_[ii], z=0))

        proto = arm_surface_contact_service_pb2.ArmSurfaceContactCommand(request=cmd)

        # Send the request
        robot.logger.info('Running arm surface contact...')
        arm_surface_contact_client.arm_surface_contact_command(proto)



def setup_arm_movement(config):
    """A simple example of using the Boston Dynamics API to command Spot's arm and body at the same time.

    Please be aware that this demo causes the robot to walk and move its arm. You can have some
    control over how much the robot moves -- see _L_ROBOT_SQUARE and _L_ARM_CIRCLE -- but regardless, the
    robot should have at least (_L_ROBOT_SQUARE + 3) m of space in each direction when this demo is used."""

    # See hello_spot.py for an explanation of these lines.
    bosdyn.client.util.setup_logging(config.verbose)

    sdk = bosdyn.client.create_standard_sdk('OlympicCirclesSpotClient')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot requires an arm to run this example."

    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)

    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.logger.info("Powering on robot... This may take a several seconds.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        robot.logger.info("Robot powered on.")

        # Tell the robot to stand up. The command service is used to issue commands to a robot.
        robot.logger.info("Commanding robot to stand...")
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        # Unstow the arm
        # Build the unstow command using RobotCommandBuilder
        unstow = RobotCommandBuilder.arm_ready_command()

        # Issue the command via the RobotCommandClient
        unstow_command_id = command_client.robot_command(unstow)
        robot.logger.info("Unstow command issued.")

        # Wait until the stow command is successful.
        block_until_arm_arrives(command_client, unstow_command_id, 3.0)

        # Get robot pose in vision frame from robot state (we want to send commands in vision
        # frame relative to where the robot stands now)
        robot_state = robot_state_client.get_robot_state()
        vision_T_world = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)

        # In this demo, the robot will walk in a square while moving its arm in a circle.
        # There are some parameters that you can set below:

        # Initialize a robot command message, which we will build out below
        command = robot_command_pb2.RobotCommand()

        # points in the square
        x_vals = np.array([0, 1, 1, 0, 0])
        y_vals = np.array([0, 0, 1, 1, 0])

        # Commands will be sent in the visual odometry ("vision") frame
        frame_name = VISION_FRAME_NAME

        # Build an arm trajectory by assembling points (in meters)
        # x will be the same for each point
        x = _L_ROBOT_SQUARE + 0.5

        x0 = 0
        y0 = 0

        x, y = np.where((x_[:, np.newaxis] - x0) ** 2 + (y_ - y0) ** 2 <= radius ** 2)
        # x, y = np.where((np.hypot((x_-x0)[:,np.newaxis], y_-y0)<= radius)) # alternative implementation
        for x, y in zip(x_[x], y_[y]):
            yield x, y

        _N_POINTS = x.size

        # duration in seconds for each move
        seconds_arm = _SECONDS_FULL / (_N_POINTS + 1)
        seconds_body = _SECONDS_FULL / x_vals.size

        for ii in range(_N_POINTS + 1):
            # Get coordinates relative to the robot's body
            y = (_L_ROBOT_SQUARE / 2) - _L_ARM_CIRCLE * (np.cos(2 * ii * math.pi / _N_POINTS))
            z = _VERTICAL_SHIFT + _L_ARM_CIRCLE * (np.sin(2 * ii * math.pi / _N_POINTS))

            # Using the transform we got earlier, transform the points into the world frame
            x_ewrt_vision, y_ewrt_vision, z_ewrt_vision = vision_T_world.transform_point(
                x, y, z)

            # Add a new point to the robot command's arm cartesian command se3 trajectory
            # This will be an se3 trajectory point
            point = command.synchronized_command.arm_command.arm_cartesian_command.pose_trajectory_in_task.points.add(
            )

            # Populate this point with the desired position, rotation, and duration information
            point.pose.position.x = x_ewrt_vision
            point.pose.position.y = y_ewrt_vision
            point.pose.position.z = z_ewrt_vision

            point.pose.rotation.x = vision_T_world.rot.x
            point.pose.rotation.y = vision_T_world.rot.y
            point.pose.rotation.z = vision_T_world.rot.z
            point.pose.rotation.w = vision_T_world.rot.w

            traj_time = (ii + 1) * seconds_arm
            duration = seconds_to_duration(traj_time)
            point.time_since_reference.CopyFrom(duration)

        # set the frame for the hand trajectory
        command.synchronized_command.arm_command.arm_cartesian_command.root_frame_name = frame_name

        # Build a body se2trajectory by first assembling points
        for ii in range(x_vals.size):
            # Pull the point in the square relative to the robot and scale according to param
            x = _L_ROBOT_SQUARE * x_vals[ii]
            y = _L_ROBOT_SQUARE * y_vals[ii]

            # Transform desired position into world frame
            x_ewrt_vision, y_ewrt_vision, z_ewrt_vision = vision_T_world.transform_point(
                x, y, 0)

            # Add a new point to the robot command's arm cartesian command se3 trajectory
            # This will be an se2 trajectory point
            point = command.synchronized_command.mobility_command.se2_trajectory_request.trajectory.points.add(
            )

            # Populate this point with the desired position, angle, and duration information
            point.pose.position.x = x_ewrt_vision
            point.pose.position.y = y_ewrt_vision

            point.pose.angle = vision_T_world.rot.to_yaw()

            traj_time = (ii + 1) * seconds_body
            duration = seconds_to_duration(traj_time)
            point.time_since_reference.CopyFrom(duration)

        # set the frame for the body trajectory
        command.synchronized_command.mobility_command.se2_trajectory_request.se2_frame_name = frame_name

        # Constrain the robot not to turn, forcing it to strafe laterally.
        speed_limit = SE2VelocityLimit(max_vel=SE2Velocity(linear=Vec2(x=2, y=2), angular=0),
                                       min_vel=SE2Velocity(linear=Vec2(x=-2, y=-2), angular=0))
        mobility_params = spot_command_pb2.MobilityParams(vel_limit=speed_limit)

        command.synchronized_command.mobility_command.params.CopyFrom(
            RobotCommandBuilder._to_any(mobility_params))

        # Send the command using the command client
        # The SE2TrajectoryRequest requires an end_time, which is set
        # during the command client call
        robot.logger.info("Sending arm and body trajectory commands.")
        command_client.robot_command(command, end_time_secs=time.time() + _SECONDS_FULL)
        time.sleep(_SECONDS_FULL + 2)

        # Power the robot off. By specifying "cut_immediately=False", a safe power off command
        # is issued to the robot. This will attempt to sit the robot before powering off.
        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), "Robot power off failed."
        robot.logger.info("Robot safely powered off.")


def main(argv):
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)
    try:
        surface_contact()
        #setup_arm_movement(options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)
