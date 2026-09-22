"""Send a fake car pose so TouchDesigner and the simulation can be tested
without the Kinect.

    python -m tracking.fake_tracker            # car spins slowly in place
    python -m tracking.fake_tracker --still    # car holds still at the origin (calibration)
"""

import argparse
import math
import time

from pythonosc.udp_client import SimpleUDPClient

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", action="store_true")
    args = ap.parse_args()

    outputs = [SimpleUDPClient(config.OSC_HOST, p) for p in (config.TD_PORT, config.SIM_PORT)]
    print("Sending fake poses. Ctrl+C to stop.")
    start = time.time()
    try:
        while True:
            yaw = 0.0 if args.still else (20 * (time.time() - start)) % 360  # 20°/s
            pose = [0.0, 0.0, 0.0, 0.0, yaw, 0.0]
            for out in outputs:
                out.send_message(config.OSC_ADDRESS, pose)
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
