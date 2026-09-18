import cv2
import numpy as np

class RANSACHomographySolver:
    def __init__(self, reprojection_thresh=3.0):
        self.reprojection_thresh = reprojection_thresh

    def estimate_matrix(self, src_pts, dst_pts):
        if len(src_pts) < 4:
            raise ValueError("At least 4 correspondence points are required for Homography.")

        # Compute projective transformation matrix using RANSAC
        H_matrix, inlier_mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, self.reprojection_thresh
        )

        if H_matrix is None:
            raise RuntimeError("Failed to compute valid Homography matrix.")

        total_inliers = int(np.sum(inlier_mask))
        return H_matrix, inlier_mask, total_inliers
