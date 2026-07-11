"""
Headless HSV calibration helper. No display needed - run this over SSH,
then open the saved *_check.jpg files in VSCode to judge the mask.

Usage:
    python calibrate_orange.py ~/mycar/data/images/1_cam_image_array_.jpg
    python calibrate_orange.py ~/mycar/data/images/1_cam_image_array_.jpg --lower 5 100 100 --upper 20 255 255
    python calibrate_orange.py ~/mycar/data/images/1_cam_image_array_.jpg --near-row-frac 0.85 --far-row-frac 0.55

Iterate: adjust --lower/--upper, rerun, reopen the output image, repeat
until the mask (right half of the output) cleanly picks out just the
two orange lines and nothing else. The green/yellow lines drawn on the
original show exactly which rows line_follower.py will scan - if either
line lands above the floor (in furniture/walls/background), raise that
--*-row-frac value until it sits on the track surface.
"""
import argparse
import cv2
import numpy as np


def row_edges(mask, row):
    """Same left/right edge detection logic as LineFollower._row_edges."""
    w = mask.shape[1]
    cols = np.where(mask[row] > 0)[0]
    if len(cols) == 0:
        return None, None
    center = w / 2.0
    left_cols = cols[cols < center]
    right_cols = cols[cols >= center]
    left = float(left_cols.max()) if len(left_cols) else None
    right = float(right_cols.min()) if len(right_cols) else None
    return left, right


def report_row(name, mask, y, w):
    left, right = row_edges(mask, y)
    if left is not None and right is not None:
        width_px = right - left
        print(f"{name} row (y={y}): left={left:.0f} right={right:.0f} "
              f"width_px={width_px:.0f} width_frac={width_px / w:.3f}")
    elif left is not None:
        print(f"{name} row (y={y}): only LEFT edge visible at x={left:.0f}")
    elif right is not None:
        print(f"{name} row (y={y}): only RIGHT edge visible at x={right:.0f}")
    else:
        print(f"{name} row (y={y}): no orange detected")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to a sample frame, e.g. from mycar/data/images/")
    ap.add_argument("--lower", type=int, nargs=3, default=[5, 100, 100], metavar=("H", "S", "V"))
    ap.add_argument("--upper", type=int, nargs=3, default=[20, 255, 255], metavar=("H", "S", "V"))
    ap.add_argument("--near-row-frac", type=float, default=0.85,
                     help="fraction down the image of the near scan row (should match LineFollower)")
    ap.add_argument("--far-row-frac", type=float, default=0.55,
                     help="fraction down the image of the far scan row (should match LineFollower)")
    ap.add_argument("--out", default="hsv_check.jpg")
    args = ap.parse_args()

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise SystemExit(f"could not read image: {args.image}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(args.lower), np.array(args.upper))
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    annotated = img_bgr.copy()
    near_y = int(h * args.near_row_frac)
    far_y = int(h * args.far_row_frac)
    cv2.line(annotated, (0, near_y), (w, near_y), (0, 255, 0), 2)   # near row = green
    cv2.line(annotated, (0, far_y), (w, far_y), (0, 255, 255), 2)   # far row = yellow
    cv2.line(mask_bgr, (0, near_y), (w, near_y), (0, 255, 0), 2)
    cv2.line(mask_bgr, (0, far_y), (w, far_y), (0, 255, 255), 2)

    side_by_side = np.hstack([annotated, mask_bgr])
    cv2.imwrite(args.out, side_by_side)
    print(f"lower={args.lower} upper={args.upper}")
    print(f"near row (green) at y={near_y}, far row (yellow) at y={far_y}, image height={h}, image width={w}")
    report_row("near", mask, near_y, w)
    report_row("far", mask, far_y, w)
    print(f"wrote {args.out} (left: original, right: mask) - open it in VSCode")
    print(f"orange pixels detected: {int((mask > 0).sum())}")


if __name__ == "__main__":
    main()
