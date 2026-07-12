import os
import cv2
import numpy as np


class LineFollower:
    """
    DonkeyCar Part: follows an oval track bounded by two orange lines.
    Drop-in replacement for the trained Keras/tflite pilot Part - same
    inputs/outputs contract ('cam/image_array' -> 'pilot/angle', 'pilot/throttle').
    """

    def __init__(self,
                 hsv_lower=(5, 60, 60),
                 hsv_upper=(20, 255, 255),
                 brightness_roi_frac=0.45,
                 near_row_frac=0.75,
                 far_row_frac=0.55,
                 near_lane_width_frac=0.58,
                 far_lane_width_frac=0.29,
                 steering_kp=2.8,
                 steering_kd=0.5,
                 error_smoothing=0.85,
                 max_throttle=0.5,
                 min_throttle=0.18,
                 curve_throttle_gain=4.0,
                 max_lost_frames=25,
                 debug=True):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.brightness_roi_frac = brightness_roi_frac
        self.near_row_frac = near_row_frac
        self.far_row_frac = far_row_frac
        self.near_lane_width_frac = near_lane_width_frac
        self.far_lane_width_frac = far_lane_width_frac
        self.kp = steering_kp
        self.kd = steering_kd
        self.smoothing = error_smoothing
        self.max_throttle = max_throttle
        self.min_throttle = min_throttle
        self.curve_gain = curve_throttle_gain
        self.max_lost_frames = max_lost_frames
        self.debug = debug
        self.denoise_kernel = np.ones((3, 3), np.uint8)
        self.debug_dir = "debug_frames"
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)

        self.prev_error = 0.0
        self.smoothed_error = 0.0
        self.last_steering = 0.0
        self.lost_frames = 0
        self.frame_count = 0

    def _normalize_brightness(self, hsv):
        """Stretch the V channel of the floor ROI to fill 0-255 each frame, so the same
        HSV threshold works whether the room is dim or bright - fixes lighting only,
        color (hue) is untouched."""
        h = hsv.shape[0]
        roi_top = int(h * self.brightness_roi_frac)
        roi_v = hsv[roi_top:, :, 2]
        lo, hi = np.percentile(roi_v, [2, 98])
        if hi - lo < 10:
            return hsv
        v = hsv[:, :, 2].astype(np.float32)
        v = np.clip((v - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
        hsv[:, :, 2] = v
        return hsv

    def _row_edges(self, mask, row):
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

    def _lane_center(self, mask, row_frac, lane_width_frac, img_w):
        row = int(mask.shape[0] * row_frac)
        left, right = self._row_edges(mask, row)
        lane_w = img_w * lane_width_frac
        if left is not None and right is not None:
            center = (left + right) / 2.0
        elif left is not None:
            # only one edge visible (turn): offset from it by half the lane width at this row's depth
            center = left + lane_w / 2.0
        elif right is not None:
            center = right - lane_w / 2.0
        else:
            center = None
        return center, left, right, row

    def _save_debug_frame(self, img_arr, mask, near_y, near_l, near_r, far_y, far_l, far_r,
                           error, smoothed_error, steering, throttle, curvature):
        annotated = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR).copy()
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for img in (annotated, mask_bgr):
            cv2.line(img, (0, near_y), (img.shape[1], near_y), (0, 255, 0), 1)
            cv2.line(img, (0, far_y), (img.shape[1], far_y), (0, 255, 255), 1)
        for x, y, color in [(near_l, near_y, (255, 0, 0)), (near_r, near_y, (0, 0, 255)),
                             (far_l, far_y, (255, 0, 0)), (far_r, far_y, (0, 0, 255))]:
            if x is not None:
                cv2.circle(annotated, (int(x), y), 4, color, -1)
                cv2.circle(mask_bgr, (int(x), y), 4, color, -1)

        combined = np.hstack([annotated, mask_bgr])
        text = (f"steer={steering:+.2f} throttle={throttle:.2f} curvature={curvature:.2f} "
                f"error={error:+.2f} smoothed={smoothed_error:+.2f}")
        cv2.putText(combined, text, (5, combined.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # rotating buffer so the SD card doesn't fill up - keeps the last ~30 saved frames
        slot = (self.frame_count // 10) % 30
        path = os.path.join(self.debug_dir, f"frame_{slot:02d}.jpg")
        cv2.imwrite(path, combined)
        # overwritten every save (~2x/sec) - open this one file in VSCode and leave it open
        # for a near-live view; VSCode's image preview auto-refreshes on file change
        cv2.imwrite(os.path.join(self.debug_dir, "latest.jpg"), combined)

    def run(self, img_arr):
        if img_arr is None:
            return self.last_steering, self.min_throttle

        h, w = img_arr.shape[:2]
        hsv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
        hsv = self._normalize_brightness(hsv)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        # strip speckle noise so a single stray pixel can't get picked as "the" edge
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.denoise_kernel)
        # hard-exclude anything above the floor line - furniture/people/walls can never
        # be mistaken for track, no matter where the scan rows end up
        mask[:int(h * self.brightness_roi_frac), :] = 0

        near_center, near_l, near_r, near_y = self._lane_center(
            mask, self.near_row_frac, self.near_lane_width_frac, w)
        far_center, far_l, far_r, far_y = self._lane_center(
            mask, self.far_row_frac, self.far_lane_width_frac, w)

        if near_center is None and far_center is None:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                if self.debug:
                    print(f"line_follower: LOST TRACK for {self.lost_frames} frames, stopping")
                # lost the track for too long: stop rather than guess and drive off further
                return 0.0, 0.0
            # brief dropout (glare, seam in the line): hold last steering, coast slow
            return self.last_steering, self.min_throttle
        self.lost_frames = 0

        center = near_center if near_center is not None else far_center
        error = (center - w / 2.0) / (w / 2.0)
        # low-pass filter the error so mask speckle jitter can't get amplified by kd
        self.smoothed_error = self.smoothing * error + (1 - self.smoothing) * self.smoothed_error

        steering = self.kp * self.smoothed_error + self.kd * (self.smoothed_error - self.prev_error)
        steering = float(np.clip(steering, -1.0, 1.0))
        self.prev_error = self.smoothed_error
        self.last_steering = steering

        if near_center is not None and far_center is not None:
            near_err = (near_center - w / 2.0) / (w / 2.0)
            far_err = (far_center - w / 2.0) / (w / 2.0)
            curvature = abs(far_err - near_err)
        else:
            curvature = abs(error)

        throttle = self.max_throttle - self.curve_gain * curvature
        throttle = float(np.clip(throttle, self.min_throttle, self.max_throttle))

        self.frame_count += 1
        if self.debug and self.frame_count % 10 == 0:
            edges = f"near={'both' if near_center is not None and far_center is not None else 'single/none'}"
            print(f"line_follower: steer={steering:+.2f} throttle={throttle:.2f} "
                  f"curvature={curvature:.2f} {edges}")
            self._save_debug_frame(img_arr, mask, near_y, near_l, near_r, far_y, far_l, far_r,
                                    error, self.smoothed_error, steering, throttle, curvature)

        return steering, throttle
