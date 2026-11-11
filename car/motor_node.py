#!/usr/bin/env python3
"""Motor node: /cmd_vel (geometry_msgs/Twist) to the L298N via picar_drive.

Run on the Pi (picar_drive.py must be in the same folder):
    python3 motor_node.py
Optional parameters:
    python3 motor_node.py --ros-args -p drive_duty:=0.8 -p pivot_duty:=0.7 -p timeout:=0.7 -p brake_pivot:=true

Why discrete modes instead of skid-steer kinematics (measured Nov 9, 2026):
- The car does not start below duty about 0.55 to 0.6, and above that speed is
  nearly flat (0.37 to 0.45 m/s), so duty is a poor speed knob.
- Spins in place only break loose at full duty (about 70 deg/s, heavy blur).
- Pivots (inner side stopped and braked, outer side forward) turn from duty 0.7
  at about 23 deg/s, which keeps markers visible. Chosen as the turn style.

So every command becomes one of four modes. The size of v and w is ignored;
only their signs matter:
    stop      v and w both about zero
    straight  w about zero: both sides at +-drive_duty
    pivot     w nonzero: one side stopped, the other at +-pivot_duty.
              Direction of travel = sign(v) (forward if v is about zero).
              The moving side is chosen so the car turns the way w asks
              (REP-103: w > 0 is counterclockwise / left).
              The stopped side is braked (brake_pivot), not left to coast.

The node also publishes /cmd_applied (Twist): the nominal measured motion of
the mode actually sent to the motors (v about 0.42 m/s straight, w about
0.40 rad/s pivoting). The filters only use its sign and mode; their forward
speed (0.40 m/s) was measured later from marker ranges (pipeline/speed_check.py). Recorded in bags, it is the honest control input for the
EKF predict step, instead of whatever magnitude the keyboard asked for.
"""
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from picar_drive import Drive


class MotorNode(Node):
    def __init__(self, drive):
        super().__init__("motor_node")
        self.drive = drive
        p = self.declare_parameter
        self.drive_duty = p("drive_duty", 0.8).value      # straight: about 0.42 m/s
        self.pivot_duty = p("pivot_duty", 0.7).value      # pivot: about 23 deg/s braked
        self.v_nominal = p("v_nominal", 0.42).value       # m/s at drive_duty (Oct 25)
        self.w_pivot = p("w_pivot", 0.40).value          # rad/s, braked pivot at 0.7 (about 23 deg/s, Nov 9)
        self.eps = p("deadzone", 0.01).value              # |v| or |w| below this = zero
        self.timeout = p("timeout", 0.7).value            # watchdog (s)
        self.brake_pivot = p("brake_pivot", True).value   # hold the inner side in pivots

        self.last_cmd = time.monotonic()
        self.mode = "stop"
        self.pub = self.create_publisher(Twist, "cmd_applied", 10)
        self.create_subscription(Twist, "cmd_vel", self.on_cmd, 10)
        self.create_timer(0.1, self.watchdog)
        self.get_logger().info(
            f"ready: drive_duty={self.drive_duty}, pivot_duty={self.pivot_duty}, "
            f"timeout={self.timeout} s, brake_pivot={self.brake_pivot}")

    def decide(self, v, w):
        """Returns (mode name, left duty, right duty, nominal v, nominal w)."""
        moving_v = abs(v) > self.eps
        turning = abs(w) > self.eps
        if not moving_v and not turning:
            return "stop", 0.0, 0.0, 0.0, 0.0
        if not turning:
            d = math.copysign(self.drive_duty, v)
            vn = math.copysign(self.v_nominal, v)
            return ("forward" if v > 0 else "back"), d, d, vn, 0.0
        # Pivot. The moving side travels in direction s (+1 forward, -1 back).
        # Right side moving forward turns the car left (w > 0); left side
        # moving forward turns it right. Reversing flips both.
        s = 1.0 if (v > self.eps or not moving_v) else -1.0
        duty = s * self.pivot_duty
        wn = math.copysign(self.w_pivot, w)
        name = f"pivot {'left' if w > 0 else 'right'}{'' if s > 0 else ' (reverse)'}"
        if w * s > 0:
            return name, 0.0, duty, 0.0, wn
        return name, duty, 0.0, 0.0, wn

    def publish_applied(self, vn, wn):
        t = Twist()
        t.linear.x = vn
        t.angular.z = wn
        self.pub.publish(t)

    def on_cmd(self, msg):
        mode, left, right, vn, wn = self.decide(msg.linear.x, msg.angular.z)
        self.drive.set(left, right, brake=self.brake_pivot and mode.startswith("pivot"))
        self.last_cmd = time.monotonic()
        self.publish_applied(vn, wn)
        if mode != self.mode:  # log changes only (teleop repeats commands)
            self.get_logger().info(f"{mode}: duty L={left:+.2f} R={right:+.2f}")
            self.mode = mode

    def watchdog(self):
        if self.mode != "stop" and time.monotonic() - self.last_cmd > self.timeout:
            self.drive.stop()
            self.publish_applied(0.0, 0.0)
            self.mode = "stop"
            self.get_logger().info("watchdog: no command, motors stopped")


def main():
    rclpy.init()
    with Drive() as drive:
        node = MotorNode(drive)
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            drive.stop()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
