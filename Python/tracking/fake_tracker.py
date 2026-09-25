'''fake_tracker.py: send made-up poses so the rest of the pipeline can be tested.

Run: python -m tracking.fake_tracker          (model spins slowly in place)
     python -m tracking.fake_tracker --still  (model holds at the origin)

Sends the same OSC message as KinectReader, to the same two ports, so
TouchDesigner and the simulation cannot tell the difference. Luke, Ana, you can use this to test your stuff without a Kinect.

Btw, this file uses the config.py constants, so if you change the ports or the OSC address, change them there too (and let me know :) ).
'''

import argparse
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
            pose = [0.0, 0.0, 0.0, 0.0, yaw, 0.0]  # since only yaw is rotating, the model will turn left and right
            for out in outputs:
                out.send_message(config.OSC_ADDRESS, pose)
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
