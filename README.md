<div align="center">

# 🖼️ Image Panorama Stitcher

### Modular, High-Performance Computer Vision Pipeline

*Automatically aligns, warps, and stitches overlapping images into seamless wide-angle panoramas — without visible seam lines or dark vignetting artifacts.*

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-cv2-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Feature%20Matching](https://img.shields.io/badge/Matching-SIFT%20%2B%20FLANN-00BFFF?style=flat)
![Homography](https://img.shields.io/badge/Geometry-RANSAC-F97316?style=flat)
![License](https://img.shields.io/badge/License-MIT-4C1?style=flat)
![Status](https://img.shields.io/badge/Status-Active-4C1?style=flat)

</div>

---

## ⚙️ How It Works

```
  [📸 Left Image]  \
                    ├─► [🔍 SIFT Feature Extraction] ──► [🤝 FLANN Nearest Matching]
  [📸 Right Image] /                                             │
                                                                  ▼
  [🖼️ Output Panorama] ◄── [🎨 Distance-Transform Blend] ◄── [📐 RANSAC Homography]
```

1. **Feature Detection** — Extracts invariant SIFT keypoints and local descriptor vectors.
2. **Keypoint Matching** — Computes nearest neighbor matches using FLANN matcher filtered via Lowe's Ratio Test (0.75).
3. **Robust Alignment** — Computes perspective transformation matrix using RANSAC outlier rejection.
4. **Canvas Expansion** — Dynamically warps images onto an expanded bounding coordinate box.
5. **Seamless Feathering** — Applies Euclidean Distance Transform for smooth linear alpha blending across overlapping regions.

---

## ✨ Key Features & Capabilities

| Area | Feature | Description |
|---|---|---|
| 🔍 Feature Extraction | SIFT Algorithm | Scale-invariant feature detection resistant to rotation and illumination changes. |
| 🎯 Matching Engine | FLANN + Ratio Test | Fast KD-Tree nearest neighbor search using Lowe's 0.75 distance threshold. |
| 📐 Geometry Solver | RANSAC Homography | Robust 3x3 transformation estimation that handles outliers effortlessly. |
| 🎨 Seam Elimination | Alpha Feathering | Distance-Transform weight mapping that completely removes harsh boundary lines. |
| 🏗️ Architecture | Modular OOP Package | Modular structure (`src/`) following clean software engineering principles. |
| 🧪 Instant Testing | Built-in CLI Demo | `--demo` flag creates synthetic test patterns without requiring external dataset downloads. |

---

## ⚡ Quick Start

### 1️⃣ Clone & Install

```bash
# Clone repository
git clone https://github.com/Priyanshi-2355/image-panorama-stitcher.git
cd image-panorama-stitcher

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Run Demo Mode

Test the pipeline instantly with synthetic overlapping images:

```bash
python main.py --demo
```

### 3️⃣ Stitch Custom Images

Pass your own image paths to generate a panorama:

```bash
python main.py --left dataset/left.jpg --right dataset/right.jpg
```

---

## 🧩 Project Architecture

```
image-panorama-stitcher/
├── 📁 src/                     # Core Processing Package
│   ├── 📄 __init__.py          # Package Initializer
│   ├── 📄 features.py          # SIFTFeatureMatcher (Keypoints & FLANN matching)
│   ├── 📄 homography.py        # RANSACHomographySolver (Matrix estimation)
│   └── 📄 stitcher.py          # PanoramaBlender (Feather blending engine)
├── 📄 main.py                  # CLI Driver Execution script
├── 📄 requirements.txt         # Project dependencies
└── 📄 README.md                # Project documentation
```

---

## 🎛️ CLI Configuration Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--demo` | flag | `False` | Runs the pipeline using internally generated synthetic images. |
| `--left` | string | `None` | File path to the left-side source image. |
| `--right` | string | `None` | File path to the right-side source image. |
| `--out_dir` | string | `"outputs"` | Output folder where `panorama_result.png` will be saved. |

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Computer Vision:** OpenCV (`cv2`)
- **Numerical Processing:** NumPy

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.
