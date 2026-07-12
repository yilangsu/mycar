"""
Update the orange-line debug view to show RAW camera (left) | DETECTION (right)
side by side in the browser at http://<car-ip>:8891/

Run once from your car folder (safe to run more than once):

    python3 dbgpatch2.py

Then restart:  python manage.py drive
"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
lf = os.path.join(BASE, "parts", "line_follower.py")
src = open(lf).read()

NEW = '''    def debug_overlay(self, img_arr):
        """Return the raw camera frame and the CV view side by side (left = raw,
        right = detection). Streamed to the browser so you can compare what the
        camera sees with what the code detects (magenta = detected orange)."""
        if img_arr is None:
            return None

        raw = img_arr.copy()
        overlay = img_arr.copy()
        h, w = overlay.shape[:2]
        hsv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        overlay[mask > 0] = (255, 0, 255)  # tint detected orange pixels magenta

        cx = w // 2
        cv2.line(overlay, (cx, 0), (cx, h), (255, 255, 255), 1)  # image center

        centers = {}
        for name, frac, lwf, color in (
                ("near", self.near_row_frac, self.near_lane_width_frac, (0, 255, 0)),
                ("far", self.far_row_frac, self.far_lane_width_frac, (255, 255, 0))):
            row = int(h * frac)
            cv2.line(overlay, (0, row), (w, row), color, 1)
            left, right = self._row_edges(mask, row)
            if left is not None:
                cv2.circle(overlay, (int(left), row), 4, color, -1)
            if right is not None:
                cv2.circle(overlay, (int(right), row), 4, color, -1)
            lc = self._lane_center(mask, frac, lwf, w)
            if lc is not None:
                cv2.circle(overlay, (int(lc), row), 6, (255, 0, 0), 2)  # lane center = red
            centers[name] = lc

        near_lc, far_lc = centers["near"], centers["far"]
        n_orange = int((mask > 0).sum())
        if near_lc is None and far_lc is None:
            status, stcolor = "NO LINES", (255, 0, 0)
        else:
            c = near_lc if near_lc is not None else far_lc
            err = (c - w / 2.0) / (w / 2.0)
            status, stcolor = ("err=%+.2f" % err), (0, 255, 0)

        cv2.putText(raw, "CAMERA", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(overlay, "orange px:%d" % n_orange, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(overlay, status, (5, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, stcolor, 1)
        return np.hstack([raw, overlay])

'''

run_anchor = "    def run(self, img_arr):"
if run_anchor not in src:
    print("ERROR: could not find 'def run' in line_follower.py - not changed")
    sys.exit(1)

if "def debug_overlay" in src:
    start = src.index("    def debug_overlay")
    end = src.index(run_anchor)
    src = src[:start] + NEW + src[end:]
    print("line_follower.py: debug_overlay updated to side-by-side")
else:
    src = src.replace(run_anchor, NEW + run_anchor, 1)
    print("line_follower.py: debug_overlay added (side-by-side)")

open(lf, "w").write(src)
print("DONE")
