import cv2
import numpy as np

class PanoramaBlender:
    @staticmethod
    def warp_and_feather_blend(left_img, right_img, H_matrix):
        h_a, w_a = left_img.shape[:2]
        h_b, w_b = right_img.shape[:2]

        # Determine transformed boundary corners
        corner_pts_b = np.float32([[0, 0], [0, h_b], [w_b, h_b], [w_b, 0]]).reshape(-1, 1, 2)
        warped_corners_b = cv2.perspectiveTransform(corner_pts_b, H_matrix)

        corner_pts_a = np.float32([[0, 0], [0, h_a], [w_a, h_a], [w_a, 0]]).reshape(-1, 1, 2)
        all_corners = np.concatenate((corner_pts_a, warped_corners_b), axis=0)

        # Calculate expanded bounding box coordinates
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        shift_x, shift_y = -x_min, -y_min
        translation_matrix = np.array([
            [1, 0, shift_x],
            [0, 1, shift_y],
            [0, 0, 1]
        ])

        canvas_dims = (x_max - x_min, y_max - y_min)

        # Warp target perspective
        warped_b = cv2.warpPerspective(right_img, translation_matrix.dot(H_matrix), canvas_dims)

        # Place anchor image on expanded canvas
        warped_a = np.zeros_like(warped_b)
        warped_a[shift_y : shift_y + h_a, shift_x : shift_x + w_a] = left_img

        # Extract region masks for valid content
        mask_a = (cv2.cvtColor(warped_a, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
        mask_b = (cv2.cvtColor(warped_b, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)

        # Euclidean distance maps for linear alpha calculation
        dist_map_a = cv2.distanceTransform(mask_a, cv2.DIST_L2, 5)
        dist_map_b = cv2.distanceTransform(mask_b, cv2.DIST_L2, 5)

        # Generate smooth transition weight matri
        sum_distances = dist_map_a + dist_map_b
        weight_alpha = np.zeros_like(dist_map_a, dtype=np.float32)
        overlapping_region = sum_distances > 0
        weight_alpha[overlapping_region] = dist_map_a[overlapping_region] / sum_distances[overlapping_region]

        blend_weights = np.repeat(weight_alpha[:, :, np.newaxis], 3, axis=2)

        # Weighted linear alpha blending
        stitched_result = (warped_a.astype(np.float32) * blend_weights) + \
                          (warped_b.astype(np.float32) * (1.0 - blend_weights))

        return np.clip(stitched_result, 0, 255).astype(np.uint8)
