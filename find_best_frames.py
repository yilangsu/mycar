"""
Scans every recorded image and scores how well the orange track lines are
detected in each one, so you can pick good frames to calibrate against
instead of guessing filenames by hand.

Usage:
    python find_best_frames.py data/images
    python find_best_frames.py data/images --lower 5 100 100 --upper 20 255 255

Run this any time lighting changes (e.g. after closing the blinds) to find
fresh representative frames - both "both edges visible" (straights, best for
lane-width calibration) and "single edge visible" (turns).
"""
import argparse
import glob
import os
import cv2
import numpy as np


def normalize_brightness(hsv, roi_top_frac):
    """Same brightness normalization as LineFollower._normalize_brightness."""
    h = hsv.shape[0]
    roi_top = int(h * roi_top_frac)
    roi_v = hsv[roi_top:, :, 2]
    lo, hi = np.percentile(roi_v, [2, 98])
    if hi - lo < 10:
        return hsv
    v = hsv[:, :, 2].astype(np.float32)
    v = np.clip((v - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    hsv[:, :, 2] = v
    return hsv


def row_edges(mask, row):
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


def classify(mask, near_y, far_y):
    near_l, near_r = row_edges(mask, near_y)
    far_l, far_r = row_edges(mask, far_y)
    near_both = near_l is not None and near_r is not None
    far_both = far_l is not None and far_r is not None
    near_any = near_l is not None or near_r is not None
    far_any = far_l is not None or far_r is not None

    if near_both and far_both:
        category = "both_near_far"
    elif near_both:
        category = "near_both_only"
    elif near_any or far_any:
        category = "single_edge"
    else:
        category = "none"
    return category


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="directory of images, e.g. data/images")
    ap.add_argument("--lower", type=int, nargs=3, default=[5, 60, 60], metavar=("H", "S", "V"))
    ap.add_argument("--upper", type=int, nargs=3, default=[20, 255, 255], metavar=("H", "S", "V"))
    ap.add_argument("--near-row-frac", type=float, default=0.75)
    ap.add_argument("--far-row-frac", type=float, default=0.55)
    ap.add_argument("--brightness-roi-frac", type=float, default=0.45)
    ap.add_argument("--top", type=int, default=8, help="how many frames to list per category")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "*.jpg")))
    if not paths:
        raise SystemExit(f"no .jpg files found in {args.dir}")

    lower = np.array(args.lower)
    upper = np.array(args.upper)

    buckets = {"both_near_far": [], "near_both_only": [], "single_edge": [], "none": []}

    for p in paths:
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        hsv = normalize_brightness(hsv, args.brightness_roi_frac)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        near_y = int(h * args.near_row_frac)
        far_y = int(h * args.far_row_frac)
        category = classify(mask, near_y, far_y)
        pixel_count = int((mask > 0).sum())
        buckets[category].append((p, pixel_count))

    print(f"scanned {len(paths)} images with lower={args.lower} upper={args.upper}\n")
    for category in ["both_near_far", "near_both_only", "single_edge", "none"]:
        items = buckets[category]
        print(f"{category}: {len(items)} frames")
        if category == "none":
            continue
        # prefer frames with a moderate, non-noisy pixel count: sort by count, take a spread
        items_sorted = sorted(items, key=lambda x: x[1])
        mid = len(items_sorted) // 2
        spread = items_sorted[max(0, mid - args.top // 2): mid + args.top // 2] or items_sorted[:args.top]
        for path, count in spread[:args.top]:
            print(f"  {path}  (orange_pixels={count})")
        print()


if __name__ == "__main__":
    main()
