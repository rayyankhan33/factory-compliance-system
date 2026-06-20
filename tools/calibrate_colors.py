"""
Small calibration helper: sample actual HSV pixel values from a real clip so
you can sanity-check (or correct) the hard-coded ranges in
src/detection/color_utils.py's CONFIG dict.

Typical flow:
    # 1. Dump a frame to look at and pick rough pixel coordinates for, say,
    #    a person's vest, or the green walkway paint:
    python tools/calibrate_colors.py data/some_clip.mp4 --frame 30 --save-frame /tmp/frame30.png

    # (open /tmp/frame30.png, note the bounding box of the region you care about)

    # 2. Sample the HSV stats inside that box:
    python tools/calibrate_colors.py data/some_clip.mp4 --frame 30 --bbox 120,80,260,400

    # Compare the printed H/S/V min/max/mean against the relevant range in
    # color_utils.py's CONFIG (e.g. green_hsv_lower/upper for a green vest or
    # walkway marking) and widen/narrow as needed.
"""
import argparse
import sys

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_path")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to sample")
    parser.add_argument("--bbox", default=None, help="x1,y1,x2,y2 -- region to sample. Omit to sample the whole frame.")
    parser.add_argument("--save-frame", default=None, help="If set, save this frame as a PNG so you can pick bbox coordinates visually.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.clip_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"Could not read frame {args.frame} from {args.clip_path}", file=sys.stderr)
        sys.exit(1)

    if args.save_frame:
        cv2.imwrite(args.save_frame, frame)
        print(f"Saved frame {args.frame} -> {args.save_frame}")

    if args.bbox:
        x1, y1, x2, y2 = [int(v) for v in args.bbox.split(",")]
        region = frame[y1:y2, x1:x2]
    else:
        region = frame

    if region.size == 0:
        print("Empty region -- check your bbox coordinates.", file=sys.stderr)
        sys.exit(1)

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    print(f"\nSampled region: {region.shape[1]}x{region.shape[0]} px")
    for name, channel in (("H", h), ("S", s), ("V", v)):
        print(f"  {name}: min={channel.min():3d}  mean={channel.mean():6.1f}  max={channel.max():3d}")

    margin = 15
    h_lo, h_hi = max(0, int(h.mean()) - margin), min(179, int(h.mean()) + margin)
    s_lo = max(0, int(np.percentile(s, 10)))
    v_lo = max(0, int(np.percentile(v, 10)))
    print(f"\nSuggested CONFIG range for this sample (centered on the mean, adjust as needed):")
    print(f"  lower = ({h_lo}, {s_lo}, {v_lo})")
    print(f"  upper = ({h_hi}, 255, 255)")


if __name__ == "__main__":
    main()
