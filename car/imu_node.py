#!/usr/bin/env python3
"""IMU node: MPU-6050 on /dev/i2c-1 to sensor_msgs/Imu for bags.

Run on the Pi (mpu6050.py, the minimal I2C driver, is in the same folder):
    python3 imu_node.py

Publishes:
    /imu/data         sensor_msgs/Imu, 100 Hz, frame_id "imu_link"
    /imu/temperature  sensor_msgs/Temperature, 1 Hz (for bias vs temperature)

Notes:
- RAW rates: the gyro bias is NOT subtracted here. Each run starts with the
  car still at home for about 10 s, and the offline filter estimates the bias
  from that stretch of the bag. Raw data keeps that choice open later.
- Only angular_velocity.z is used by the planar EKF. The board lies flat with
  z up, so z is the car's yaw rate (REP-103: positive = counterclockwise).
  The board's x and y axes are NOT aligned with base_link.
- No orientation is estimated: orientation_covariance[0] = -1 (ROS convention).
- Gyro: +-500 deg/s, 44 Hz low-pass, 100 Hz output. Noise sigma 0.026 deg/s
  per sample (measured Nov 9 2026), put in angular_velocity_covariance.
- Accel: +-2 g, uncalibrated (magnitude reads about 9% high), rough covariance.
"""
import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Imu, Temperature

from mpu6050 import SMBus, s16, ADDR, BUS, WHO_AM_I, PWR_MGMT_1, CONFIG, SMPLRT_DIV, \
    GYRO_CONFIG, ACCEL_CONFIG, ACCEL_XOUT_H

GYRO_LSB = 65.5        # LSB per deg/s at +-500 deg/s
ACCEL_LSB = 16384.0    # LSB per g at +-2 g
G = 9.80665
DEG = math.pi / 180.0
GYRO_SIGMA = 0.026 * DEG   # rad/s per sample
ACCEL_SIGMA = 0.05         # m/s^2, rough guess (unused by the EKF)


class ImuNode(Node):
    def __init__(self):
        super().__init__("imu_node")
        self.rate = self.declare_parameter("rate_hz", 100.0).value
        self.frame = self.declare_parameter("frame_id", "imu_link").value

        self.bus = SMBus(BUS)
        who = self.bus.read_byte_data(ADDR, WHO_AM_I)
        self.bus.write_byte_data(ADDR, PWR_MGMT_1, 0x01)   # wake, gyro X PLL clock
        time.sleep(0.1)
        self.bus.write_byte_data(ADDR, CONFIG, 0x03)       # DLPF about 44 Hz
        self.bus.write_byte_data(ADDR, SMPLRT_DIV, 9)      # 100 Hz
        self.bus.write_byte_data(ADDR, GYRO_CONFIG, 0x08)  # +-500 deg/s
        self.bus.write_byte_data(ADDR, ACCEL_CONFIG, 0x00) # +-2 g
        time.sleep(0.1)

        self.pub = self.create_publisher(Imu, "imu/data", 50)
        self.pub_t = self.create_publisher(Temperature, "imu/temperature", 10)
        self.count = 0
        self.errors = 0
        self.create_timer(1.0 / self.rate, self.tick)
        self.get_logger().info(
            f"ready: WHO_AM_I=0x{who:02x}, {self.rate:.0f} Hz, +-500 deg/s, frame {self.frame}")

    def tick(self):
        try:
            d = self.bus.read_i2c_block_data(ADDR, ACCEL_XOUT_H, 14)
        except OSError as e:
            self.errors += 1
            if self.errors in (1, 10, 100) or self.errors % 1000 == 0:
                self.get_logger().warn(f"I2C read failed ({self.errors} so far): {e}")
            return
        stamp = self.get_clock().now().to_msg()
        ax, ay, az = (s16(d[i], d[i + 1]) / ACCEL_LSB * G for i in (0, 2, 4))
        temp = s16(d[6], d[7]) / 340.0 + 36.53
        gx, gy, gz = (s16(d[i], d[i + 1]) / GYRO_LSB * DEG for i in (8, 10, 12))

        m = Imu()
        m.header.stamp = stamp
        m.header.frame_id = self.frame
        m.orientation_covariance[0] = -1.0
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = gx, gy, gz
        m.angular_velocity_covariance[0] = m.angular_velocity_covariance[4] = \
            m.angular_velocity_covariance[8] = GYRO_SIGMA ** 2
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = ax, ay, az
        m.linear_acceleration_covariance[0] = m.linear_acceleration_covariance[4] = \
            m.linear_acceleration_covariance[8] = ACCEL_SIGMA ** 2
        self.pub.publish(m)

        self.count += 1
        if self.count % int(self.rate) == 0:
            t = Temperature()
            t.header.stamp = stamp
            t.header.frame_id = self.frame
            t.temperature = temp
            self.pub_t.publish(t)


def main():
    rclpy.init()
    node = ImuNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
