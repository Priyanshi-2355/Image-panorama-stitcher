import argparse, os, cv2, numpy as np
from src.features import SIFTFeatureMatcher
from src.homography import RANSACHomographySolver
from src.stitcher import PanoramaBlender

def generate_synthetic_pair():
    canvas = np.ones((400, 800, 3), dtype=np.uint8) * 220
    cv2.rectangle(canvas, (100, 100), (300, 300), (0, 0, 255), -1)
    cv2.circle(canvas, (400, 200), 80, (0, 255, 0), -1)
    cv2.putText(canvas, "COMPUTER VISION", (250, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

    img_left = canvas[:, :500].copy()
    img_right = canvas[:, 300:].copy()

    h, w = img_right.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 3, 1.0)
    img_right = cv2.warpAffine(img_right, M, (w, h))

    return img_left, img_right

def main():
    parser = argparse.ArgumentParser(description="Custom Automated Panorama Stitching Pipeline")
    parser.add_argument("--demo", action="store_true", help="Run stitching on synthetic pair")
    parser.add_argument("--left", help="Path to left input image")
    parser.add_argument("--right", help="Path to right input image")
    parser.add_argument("--out_dir", default="outputs", help="Output directory path")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.demo:
        print("[+] Generating overlapping synthetic test pair...")
        img_left, img_right = generate_synthetic_pair()
    elif args.left and args.right:
        img_left = cv2.imread(args.left)
        img_right = cv2.imread(args.right)
    else:
        print("Error: Specify either --demo or provide both --left and --right image paths.")
        return

    print("Extracting SIFT keypoints and computing ratio-test matches")
    matcher = SIFTFeatureMatcher()
    src_pts, dst_pts, _, _, matches = matcher.compute_and_match(img_left, img_right)
    print(f"Found {len(matches)} robust feature correspondences")

    print("Estimating Homography Matrix using RANSAC outlier rejection")
    solver = RANSACHomographySolver()
    H_matrix, inliers_mask, inlier_count = solver.estimate_matrix(src_pts, dst_pts)
    print(f"Homography matrix solved with {inlier_count} inliers")

    print("Performing perspective transformation and distance-transform alpha blending")
    panorama = PanoramaBlender.warp_and_feather_blend(img_left, img_right, H_matrix)

    out_path = os.path.join(args.out_dir, "panorama_result.png")
    cv2.imwrite(out_path, panorama)
    print(f"High-quality panorama saved to: {out_path}")

if __name__ == "__main__":
    main()
