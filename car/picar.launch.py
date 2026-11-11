"""Starts everything the car needs for a recorded drive, in one command.

Run on the Pi (the node scripts are found next to this file):
    ros2 launch car/picar.launch.py

Starts:
  camera_node       /camera/image_raw at 30 fps (focus locked), /camera/camera_info
  throttle          /camera/image_10hz: every third frame, for recording
                    (keeps the 30 fps exposure limit, so no extra motion blur,
                    while bags grow about 9 MB/s instead of 27 MB/s)
  static transform  base_link -> camera (translation Oct 24, rotation Nov 10 2026)
  imu_node.py       /imu/data at 100 Hz, /imu/temperature
  motor_node.py     /cmd_vel -> motors, /cmd_applied

Teleop runs separately in its own SSH window (it needs the keyboard).
Ctrl+C here stops everything.
"""
import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.realpath(__file__))


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="camera_ros", executable="camera_node", output="screen",
            parameters=[{
                "width": 640, "height": 360, "sensor_mode": "2304:1296",
                "AfMode": 0, "LensPosition": 1.5,
            }],
        ),
        Node(
            package="topic_tools", executable="throttle", output="screen",
            arguments=["messages", "/camera/image_raw", "12.0", "/camera/image_10hz"],
        ),
        Node(
            package="tf2_ros", executable="static_transform_publisher", output="screen",
            arguments=[
                "--x", "0.0877", "--y", "-0.0030", "--z", "0.1519",
                "--qx", "0.55827", "--qy", "-0.56189", "--qz", "0.41968", "--qw", "-0.44326",
                "--frame-id", "base_link", "--child-frame-id", "camera",
            ],
        ),
        ExecuteProcess(cmd=["python3", os.path.join(HERE, "imu_node.py")], cwd=HERE, output="screen"),
        ExecuteProcess(cmd=["python3", os.path.join(HERE, "motor_node.py")], cwd=HERE, output="screen"),
    ])
