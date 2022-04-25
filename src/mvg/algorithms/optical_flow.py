import cv2


class OpticalFlowLK:
    @staticmethod
    def track(prev_image, curr_image, points_to_track):
        tracked_points, status, error = cv2.calcOpticalFlowPyrLK(
            prevImg=prev_image, nextImg=curr_image, prevPts=points_to_track, nextPts=None
        )
        if status is None:
            raise Exception("Tracking failed.")
        return tracked_points, status.reshape(-1) == 1
