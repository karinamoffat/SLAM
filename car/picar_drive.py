#!/usr/bin/env python3
"""Skid-steer drive for the Pi car: L298N controlled with lgpio.

Use as a module:
    from picar_drive import Drive
    with Drive() as d:
        d.set(0.6, 0.6)   # left, right in [-1, 1]; positive = car moves forward
        ...
    # leaving the "with" block always stops the motors and frees the pins

Run directly for a self-test (car LIFTED):
    python3 picar_drive.py

Notes:
- gpiozero 2.0.1 on Ubuntu 26.04 forces gpiochip4 on a Pi 5, but this kernel
  exposes the header as gpiochip0, so lgpio is used directly.
- PWM is on the enable pins (drive/coast). The command is a duty cycle, not a
  speed; the duty to speed mapping must be measured on the floor.
"""
import time

import lgpio

CHIP = 0        # gpiochip0 = pinctrl-rp1, the 40-pin header
PWM_HZ = 1000   # software PWM frequency on ENA/ENB

# Channel A = RIGHT side, channel B = LEFT side (corrected Nov 9 2026: the gyro
# showed every turn mirrored, so the labels were swapped). GPIO numbers,
# physical pins in comments. "invert" belongs to the channel's motor wiring,
# so it moved with channel B (motor wires reversed, found Oct 24 2026).
SIDES = {
    "left":  {"in1": 20, "in2": 21, "en": 13, "invert": True},   # channel B, pins 38, 40, 33
    "right": {"in1": 5,  "in2": 6,  "en": 12, "invert": False},  # channel A, pins 29, 31, 32
}


class Drive:
    def __init__(self):
        self._h = lgpio.gpiochip_open(CHIP)
        self._claimed = []
        try:
            for s in SIDES.values():
                for key in ("in1", "in2", "en"):
                    lgpio.gpio_claim_output(self._h, s[key], 0)
                    self._claimed.append(s[key])
        except lgpio.error:
            self.close()
            raise

    def _side(self, name, cmd, brake=False):
        s = SIDES[name]
        cmd = max(-1.0, min(1.0, float(cmd)))
        if cmd == 0 and brake:
            # L298N "fast motor stop": enable high, both inputs low shorts the
            # motor terminals, so the wheels resist rolling instead of coasting.
            lgpio.gpio_write(self._h, s["in1"], 0)
            lgpio.gpio_write(self._h, s["in2"], 0)
            lgpio.tx_pwm(self._h, s["en"], PWM_HZ, 100.0)
            return
        if s["invert"]:
            cmd = -cmd
        if cmd > 0:
            a, b = 1, 0
        elif cmd < 0:
            a, b = 0, 1
        else:
            a, b = 0, 0
        lgpio.gpio_write(self._h, s["in1"], a)
        lgpio.gpio_write(self._h, s["in2"], b)
        lgpio.tx_pwm(self._h, s["en"], PWM_HZ, abs(cmd) * 100.0)

    def set(self, left, right, brake=False):
        """Commands in [-1, 1] per side. Positive drives the car forward.

        brake=True: a side commanded 0 is braked (held) instead of coasting.
        Used for pivots (Nov 9 2026): a coasting inner side rolls along and
        widens the turn, differently on each side.
        """
        self._side("left", left, brake)
        self._side("right", right, brake)

    def stop(self):
        self.set(0.0, 0.0)

    def close(self):
        if self._h is None:
            return
        try:
            if len(self._claimed) == 6:
                self.stop()
            for g in self._claimed:
                lgpio.gpio_write(self._h, g, 0)
                lgpio.gpio_free(self._h, g)
        except lgpio.error as e:
            print(f"Drive cleanup warning: {e}")
        lgpio.gpiochip_close(self._h)
        self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# Self-test: (description, left, right)
TESTS = [
    ("both sides FORWARD", 0.6, 0.6),
    ("both sides REVERSE", -0.6, -0.6),
    ("spin LEFT (counterclockwise seen from above)", -0.7, 0.7),
    ("spin RIGHT (clockwise seen from above)", 0.7, -0.7),
]
RUN_S = 2.0


def self_test():
    with Drive() as d:
        try:
            for desc, left, right in TESTS:
                input(f"Enter: {desc} for {RUN_S:.0f} s ")
                d.set(left, right)
                time.sleep(RUN_S)
                d.stop()
            print("Done.")
        except KeyboardInterrupt:
            print("\nInterrupted.")


if __name__ == "__main__":
    self_test()
