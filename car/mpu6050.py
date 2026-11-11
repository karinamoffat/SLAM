"""Minimal MPU-6050 access over I2C on /dev/i2c-N (Raspberry Pi 5, Ubuntu 26.04).

Shared by car/imu_node.py and the gyro experiments.
"""
import fcntl
import os

I2C_SLAVE = 0x0703  # Linux ioctl: set the device address for this file handle


class SMBus:
    """Minimal I2C access through /dev/i2c-N (no smbus2 needed).

    The apt smbus2 crashes on this Python ("SystemError: buffer overflow" in
    its I2C_FUNCS ioctl), so this talks to the kernel driver directly:
    write the register number, then read bytes. The MPU-6050 auto-increments
    the register address, so one read returns consecutive registers.
    """

    def __init__(self, bus):
        self.fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR)

    def _select(self, addr):
        fcntl.ioctl(self.fd, I2C_SLAVE, addr)

    def read_i2c_block_data(self, addr, reg, n):
        self._select(addr)
        os.write(self.fd, bytes([reg]))
        return list(os.read(self.fd, n))

    def read_byte_data(self, addr, reg):
        return self.read_i2c_block_data(addr, reg, 1)[0]

    def write_byte_data(self, addr, reg, value):
        self._select(addr)
        os.write(self.fd, bytes([reg, value & 0xFF]))


BUS = 1
ADDR = 0x68

# Registers
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B
PWR_MGMT_1 = 0x6B
WHO_AM_I = 0x75


def s16(hi, lo):
    v = (hi << 8) | lo
    return v - 65536 if v & 0x8000 else v
