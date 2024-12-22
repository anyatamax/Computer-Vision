import os

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import match_template

from detection import detection_cast, draw_detections, extract_detections
from tracker import Tracker


def gaussian(shape, x, y, dx, dy):
    """Return gaussian for tracking.

    shape: [width, height]
    x, y: gaussian center
    dx, dy: std by x and y axes

    return: numpy array (width x height) with gauss function, center (x, y) and std (dx, dy)
    """
    Y, X = np.mgrid[0 : shape[0], 0 : shape[1]]
    result = np.exp(-((X - x) ** 2) / dx**2 - (Y - y) ** 2 / dy**2)
    return result


class CorrelationTracker(Tracker):
    """Generate detections and building tracklets."""

    def __init__(self, detection_rate=5, **kwargs):
        super().__init__(**kwargs)
        self.detection_rate = detection_rate  # Detection rate
        self.prev_frame = None  # Previous frame (used in cross correlation algorithm)

    def build_tracklet(self, frame):
        """Between CNN execution uses normalized cross-correlation algorithm (match_template)."""
        detections = []
        # Write code here
        # Apply rgb2gray to frame and previous frame
        gray_frame = rgb2gray(frame)
        gray_prev_frame = rgb2gray(self.prev_frame)
        # For every previous detection
        # Use match_template + gaussian to extract detection on current frame
        for label, xmin, ymin, xmax, ymax in self.detection_history[-1]:
            # Step 0: Extract prev_bbox from prev_frame
            prev_bbox = gray_prev_frame[ymin : ymax, xmin : xmax]
            # Step 1: Extract new_bbox from current frame with the same coordinates
            new_bbox = gray_frame[ymin : ymax, xmin : xmax]
            # Step 2: Calc match_template between previous and new bbox
            # Use padding
            template_matched = match_template(new_bbox, prev_bbox, pad_input=True)
            # Step 3: Then multiply matching by gauss function
            # Find argmax(matching * gauss)
            # print(new_bbox.shape)
            template_shape_y = template_matched.shape[0]
            template_shape_x = template_matched.shape[1]
            argmax_match = np.argmax(template_matched * gaussian([template_shape_y, template_shape_x], template_shape_x // 2, template_shape_y // 2, xmax - xmin, ymax - ymin))
            ids_2d = np.unravel_index(argmax_match, (template_shape_y, template_shape_x))
            index_0 = ids_2d[0] - template_matched.shape[0] // 2 + ymin
            index_1 = ids_2d[1] - template_matched.shape[1] // 2 + xmin
            # # Step 4: Append to detection list
            
            detections.append([label, np.clip(index_1, 0, frame.shape[1] - 1), np.clip(index_0, 0, frame.shape[0] - 1), np.clip(index_1 + (xmax - xmin), 0, frame.shape[1] - 1), np.clip(index_0 + (ymax - ymin), 0, frame.shape[0] - 1)])
            # print(detections[-1])

        return detection_cast(detections)

    def update_frame(self, frame):
        if not self.frame_index:
            detections = self.init_tracklet(frame)
            self.save_detections(detections)
        elif self.frame_index % self.detection_rate == 0:
            detections = extract_detections(frame, labels=self.labels)
            detections = self.bind_tracklet(detections)
            self.save_detections(detections)
        else:
            detections = self.build_tracklet(frame)

        self.detection_history.append(detections)
        self.prev_frame = frame
        self.frame_index += 1

        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


def main():
    from moviepy.editor import VideoFileClip

    dirname = os.path.dirname(__file__)
    input_clip = VideoFileClip(os.path.join(dirname, "data", "test.mp4"))

    tracker = CorrelationTracker()
    input_clip.fl_image(tracker.update_frame).preview()


if __name__ == "__main__":
    main()
