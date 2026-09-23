'''monitor.py: this watches the poses arriving over OSC.

Run: python -m tracking.monitor              (listens on 9001)
     python -m tracking.monitor --port 9000  (only if TouchDesigner is closed)

Prints position, rotation and the message rate a couple of times a second.
This is how you prove the tracker works before blaming TouchDesigner, and the
first thing to run when TouchDesigner shows nothing.

Two things only show up here: whether messages arrive at all, and how fast.
'''

import argparse
import time

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

import config

count = 0
last_print = time.time()


def show(_address, *pose):
    global count, last_print
    count += 1
    now = time.time()
    if now - last_print >= 0.5:  # twice a second is enough to read
        tx, ty, tz, rx, ry, rz = pose
        print(f"pos {tx:8.1f} {ty:8.1f} {tz:8.1f} mm   rot {rx:7.1f} {ry:7.1f} {rz:7.1f} deg"
              f"   {count / (now - last_print):5.1f} msgs/s")
        count, last_print = 0, now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=config.SIM_PORT)
    args = ap.parse_args()

    dispatcher = Dispatcher()
    dispatcher.map(config.OSC_ADDRESS, show)
    print(f"Listening on port {args.port} for {config.OSC_ADDRESS}. Ctrl+C to stop.")
    try:
        BlockingOSCUDPServer((config.OSC_HOST, args.port), dispatcher).serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
