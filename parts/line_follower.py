import cv2
import numpy as np


class LineFollower:
    """
    DonkeyCar Part: follows an oval track bounded by two orange lines.
    Drop-in replacement for the trained Keras/tflite pilot Part - same
    inputs/outputs contract ('cam/image_array' -> 'pilot/angle', 'pilot/throttle').
    """

    def __init__(self,
                 hsv_lower=(5, 100, 100),
                 hsv_upper=(20, 255, 255),
                 near_row_frac=0.75,
                 far_row_frac=0.55,
                 near_lane_width_frac=0.604,
                 far_lane_width_frac=0.304,
                 steering_kp=1.8,
                 steering_kd=0.6,
                 max_throttle=1.0,
                 min_throttle=0.35,
                 curve_throttle_gain=2.5,
                 max_lost_frames=10):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.near_row_frac = near_row_frac
        self.far_row_frac = far_row_frac
        self.near_lane_width_frac = near_lane_width_frac
        self.far_lane_width_frac = far_lane_width_frac
        self.kp = steering_kp
        self.kd = steering_kd
        self.max_throttle = max_throttle
        self.min_throttle = min_throttle
        self.curve_gain = curve_throttle_gain
        self.max_lost_frames = max_lost_frames

        self.prev_error = 0.0
        self.last_steering = 0.0
        self.lost_frames = 0

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
            return (left + right) / 2.0
        # only one edge visible (turn): offset from it by half the lane width at this row's depth
        if left is not None:
            return left + lane_w / 2.0
        if right is not None:
            return right - lane_w / 2.0
        return None

    def run(self, img_arr):
        if img_arr is None:
            return self.last_steering, self.min_throttle

        h, w = img_arr.shape[:2]
        hsv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        near_center = self._lane_center(mask, self.near_row_frac, self.near_lane_width_frac, w)
        far_center = self._lane_center(mask, self.far_row_frac, self.far_lane_width_frac, w)

        if near_center is None and far_center is None:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                # lost the track for too long: stop rather than guess and drive off further
                return 0.0, 0.0
            # brief dropout (glare, seam in the line): hold last steering, coast slow
            return self.last_steering, self.min_throttle
        self.lost_frames = 0

        center = near_center if near_center is not None else far_center
        error = (center - w / 2.0) / (w / 2.0)

        steering = self.kp * error + self.kd * (error - self.prev_error)
        steering = float(np.clip(steering, -1.0, 1.0))
        self.prev_error = error
        self.last_steering = steering

        if near_center is not None and far_center is not None:
            near_err = (near_center - w / 2.0) / (w / 2.0)
            far_err = (far_center - w / 2.0) / (w / 2.0)
            curvature = abs(far_err - near_err)
        else:
            curvature = abs(error)

        throttle = self.max_throttle - self.curve_gain * curvature
        throttle = float(np.clip(throttle, self.min_throttle, self.max_throttle))

        return steering, throttle
