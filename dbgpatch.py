"""
One-shot patch: adds a live orange-line CV debug view to this DonkeyCar app.

Run it once from inside your car folder:

    python3 dbgpatch.py

It does two things (and is safe to run more than once - it skips work already done):
  1. Adds a debug_overlay() method to parts/line_follower.py
  2. Adds an always-on browser view (port 8891) to manage.py

Then run `python manage.py drive` and open http://<car-ip>:8891/ in a browser.
"""
import os
import sys

# patch the folder this script lives in (i.e. your car folder)
BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1) Add a debug_overlay() method to parts/line_follower.py
# ---------------------------------------------------------------------------
lf_path = os.path.join(BASE, "parts", "line_follower.py")
src = open(lf_path).read()

if "def debug_overlay" in src:
    print("line_follower.py: debug_overlay already present, skipping")
else:
    method = '''    def debug_overlay(self, img_arr):
        """Return a copy of the frame annotated with what the CV sees:
        orange pixels tinted magenta, the two scan rows, the detected lane
        edges, and the computed lane center. Streamed to a browser so you can
        watch the detection live over SSH."""
        if img_arr is None:
            return None

        img = img_arr.copy()
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # tint every detected "orange" pixel magenta so it pops against the real
        # orange line - if the magenta lands on the lines, the HSV range is good.
        img[mask > 0] = (255, 0, 255)

        cx = w // 2
        cv2.line(img, (cx, 0), (cx, h), (255, 255, 255), 1)  # image center

        centers = {}
        for name, frac, lwf, color in (
                ("near", self.near_row_frac, self.near_lane_width_frac, (0, 255, 0)),
                ("far", self.far_row_frac, self.far_lane_width_frac, (255, 255, 0))):
            row = int(h * frac)
            cv2.line(img, (0, row), (w, row), color, 1)
            left, right = self._row_edges(mask, row)
            if left is not None:
                cv2.circle(img, (int(left), row), 4, color, -1)
            if right is not None:
                cv2.circle(img, (int(right), row), 4, color, -1)
            lc = self._lane_center(mask, frac, lwf, w)
            if lc is not None:
                cv2.circle(img, (int(lc), row), 6, (255, 0, 0), 2)  # lane center = red
            centers[name] = lc

        near_lc, far_lc = centers["near"], centers["far"]
        n_orange = int((mask > 0).sum())
        if near_lc is None and far_lc is None:
            status, stcolor = "NO LINES", (255, 0, 0)
        else:
            c = near_lc if near_lc is not None else far_lc
            err = (c - w / 2.0) / (w / 2.0)
            status, stcolor = ("err=%+.2f" % err), (0, 255, 0)

        cv2.putText(img, "orange px:%d" % n_orange, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(img, status, (5, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, stcolor, 1)
        return img

'''
    anchor = "    def run(self, img_arr):"
    if anchor not in src:
        print("ERROR: could not find 'def run' in line_follower.py - not patched")
        sys.exit(1)
    src = src.replace(anchor, method + anchor, 1)
    open(lf_path, "w").write(src)
    print("line_follower.py: debug_overlay added")

# ---------------------------------------------------------------------------
# 2) Add an always-on browser debug view to manage.py (port 8891)
# ---------------------------------------------------------------------------
mg_path = os.path.join(BASE, "manage.py")
m = open(mg_path).read()

if "cam/orange_debug" in m:
    print("manage.py: debug view already present, skipping")
else:
    anchor2 = ("    if cfg.USE_FPV:\n"
               "        V.add(WebFpv(), inputs=['cam/image_array'], threaded=True)\n")
    block = '''
    # --- Live orange-line CV debug view (auto-added) ---
    # Streams the camera with detected orange pixels + lane center drawn on it
    # to http://<car-ip>:8891/ so you can watch the vision work over SSH. Runs
    # in every mode, so you can drive manually and still see the detection.
    try:
        from parts.line_follower import LineFollower as _DbgLineFollower
        from donkeycar.parts.transform import Lambda as _DbgLambda
        _dbg_follower = _DbgLineFollower()
        V.add(_DbgLambda(_dbg_follower.debug_overlay),
              inputs=['cam/image_array'], outputs=['cam/orange_debug'])
        V.add(WebFpv(port=8891), inputs=['cam/orange_debug'], threaded=True)
        print(">>> Orange-line debug view at http://<car-ip>:8891/")
    except Exception as _dbg_err:
        print(">>> Orange-line debug view could not start:", _dbg_err)
'''
    if anchor2 not in m:
        print("ERROR: could not find the USE_FPV block in manage.py - not patched")
        sys.exit(1)
    m = m.replace(anchor2, anchor2 + block, 1)
    open(mg_path, "w").write(m)
    print("manage.py: debug view added")

print("DONE")
