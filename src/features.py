import cv2
import numpy as np

class SIFTFeatureMatcher:
    def __init__(self, ratio_threshold=0.75):
        # Initialize SIFT keypoint detector
        self.sift = cv2.SIFT_create()
        self.ratio_threshold = ratio_threshold

    def compute_and_match(self, left_img, right_img):
        # Convert images to grayscale for keypoint detection
        gray_a = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and calculate descriptors
        kps_a, descs_a = self.sift.detectAndCompute(gray_a, None)
        kps_b, descs_b = self.sift.detectAndCompute(gray_b, None)

        if descs_a is None or descs_b is None:
            raise ValueError("Could not extract feature descriptors from images.")

        # Fast Library for Approximate Nearest Neighbors
        flann_index_kdtree = 1
        index_params = dict(algorithm=flann_index_kdtree, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        knn_matches = flann.knnMatch(descs_b, descs_a, k=2)

        # Filter valid matches
        filtered_matches = []
        src_points, dst_points = [], []

        for m1, m2 in knn_matches:
            if m1.distance < self.ratio_threshold * m2.distance:
                filtered_matches.append(m1)
                src_points.append(kps_b[m1.queryIdx].pt)
                dst_points.append(kps_a[m1.trainIdx].pt)

        return np.float32(src_points), np.float32(dst_points), kps_b, kps_a, filtered_matches
