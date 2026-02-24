# 🏆 Kaggle Multi-Instance Object Detection - 36th Place Solution

**Competition**: Duality AI Synthetic-to-Real Challenge  
**Score**: 0.919 mAP@50 | **Rank**: 36/111 (Top 32.4%)

[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/multi-instance-object-detection-challenge)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8x-00FFFF?style=flat)](https://github.com/ultralytics/ultralytics)

---

## 📊 Competition Results

| Metric | Value |
|--------|-------|
| **Final Ranking** | **36 / 111 participants** |
| **Percentile** | **Top 32.4%** |
| **mAP@50** | **0.919** |
| **Model** | **YOLOv8x (Largest variant)** |
| **Training Images** | **89 synthetic images** |
| **Validation Images** | **10 images** |
| **Image Resolution** | **1056×1056** |

> **Key Achievement**: Reached top 33% using **minimal training data** (89 images) through strategic synthetic data generation and high-resolution model training.

---

## 🎯 Challenge Overview

### The Problem
Train an object detection model to detect **multiple soup can instances** in real-world images using **ONLY synthetic training data** generated via FalconCloud digital twin simulation.

### Key Constraints
- ❌ No real-world images for training
- ✅ Only synthetic data from FalconCloud
- 🎯 Evaluation on real-world test images (Sim2Real gap!)
- 📊 mAP@50 evaluation metric

---

## 🚀 My Approach: Quality Over Quantity

### Core Strategy
- 🎯 **89 training images** (vs. most competitors using 1000s)
- 🎯 **3 iterative refinement cycles** with FalconCloud
- 🎯 **YOLOv8x** for maximum model capacity
- 🎯 **1056px resolution** for small object detection
- 🎯 **Heavy HSV augmentation** for Sim2Real transfer

### Why This Worked
Most competitors focused on generating thousands of images. I proved that **strategic, high-quality synthetic data generation beats brute-force approaches**.

---

## 🔬 Technical Details

### 1. Synthetic Data Generation

**FalconCloud Configuration** (3 iterations):

| Parameter | Setting |
|-----------|---------|
| Training Images | 89 images |
| Validation Images | 10 images |
| Environments | Warehouse, retail, kitchen |
| Camera Distance | 0.8m - 1.5m |
| Camera Elevation | 15° - 45° |
| Occlusions | 20-40% probability |
| Instances per Image | 5-15 soup cans |
| Post-processing | Grain + blur for realism |

**Iterative Refinement Process**:
1. **Phase 1**: Baseline dataset → Identified occlusion weaknesses
2. **Phase 2**: Added realistic obstructions → Improved lighting diversity
3. **Phase 3**: Fine-tuned camera parameters → Matched real-world aesthetics

### 2. Model Configuration

**Base Model**: YOLOv8x (Largest variant)

```python
from ultralytics import YOLO

model = YOLO('yolov8x.pt')
```

**Training Hyperparameters**:
```python
model.train(
    data='data/data.yaml',
    epochs=300,
    batch=4,
    imgsz=1056,
    patience=300,
    optimizer='SGD',
    momentum=0.937,
    lr0=0.001,
    weight_decay=0.0005,
    cos_lr=True,
    save_period=2,
    workers=8,
    # Augmentations
    close_mosaic=20,
    hsv_h=0.015,
    hsv_s=0.7,      # Heavy saturation variation
    hsv_v=0.4,      # Brightness variation
    flipud=0,       # No vertical flips
    fliplr=0.5,     # 50% horizontal flip
    translate=0.1,
    scale=0.5,
    shear=0,
    warmup_epochs=3,
    warmup_momentum=1,
)
```

### 3. Key Training Decisions

**Why 1056px Image Size?**
- Soup cans are small objects
- Higher resolution preserves fine details
- Critical for multi-instance detection with occlusions
- Trade-off: 4× GPU memory, but worth the accuracy gain

**Why YOLOv8x?**
- Maximum model capacity with limited data
- Better feature learning with only 89 images
- Pretrained COCO weights provide strong baseline

**Why Heavy HSV Augmentation?**
- HSV_S=0.7 allows significant color variation
- Bridges Sim2Real lighting distribution gap
- Compensates for synthetic data's "perfect" appearance

**Training Results (Epoch 288/300)**:
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances   Size
288/300   6.64G     0.141      0.1177     0.7799    10          1056
```

---

## 📁 Repository Structure

```
kaggle-synthetic-object-detection/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── data/                              # Dataset
│   ├── train/
│   │   ├── images/                    # 89 training images
│   │   └── labels/                    # YOLO format labels
│   ├── val/
│   │   ├── images/                    # 10 validation images
│   │   └── labels/                    # YOLO format labels
│   └── data.yaml                      # Dataset configuration
│
├── scripts/                           # Python scripts
│   ├── train.py                       # Training script
│   ├── predict.py                     # Inference script
│   └── visualize.py                   # Visualization script
│
├── results/                           # Competition results
│   ├── submission.csv                 # Final submission (0.919 mAP)
│   └── training_curves.png            # Training progress
│
└── runs/                              # YOLOv8 outputs (git-ignored)
    └── train/exp/weights/best.pt      # Best model checkpoint
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/SharifNirjon/kaggle-synthetic-object-detection.git
cd kaggle-synthetic-object-detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Training

```python
python scripts/train.py
```

### Inference

```python
python scripts/predict.py --weights runs/train/exp/weights/best.pt --source test_images/
```

---

## 💡 Key Learnings

### What Worked ✅

1. **Quality > Quantity in Synthetic Data**
   - 89 strategic images beat thousands of random images
   - Iterative refinement based on model feedback

2. **High-Resolution Training (1056px)**
   - Critical for small object detection
   - Worth the GPU memory trade-off

3. **YOLOv8x with Limited Data**
   - Model capacity compensated for dataset size
   - Pretrained weights + fine-tuning worked well

4. **Heavy HSV Augmentation (S=0.7, V=0.4)**
   - Bridged Sim2Real color distribution gap
   - Essential for domain adaptation

5. **No Vertical Flips**
   - Soup cans have natural orientation
   - Domain-aware augmentation choices matter

### Challenges ⚠️

1. **Limited FalconCloud Credits** → Only 89 images possible
2. **Sim2Real Color Gap** → Solved with HSV augmentation
3. **GPU Memory Constraints** → Batch size=4 compromise
4. **Small Object Detection** → 1056px resolution required

### Future Improvements 🔄

- Generate 200-300 images with more credits
- Ensemble multiple models (YOLOv8x + YOLOv9)
- Test-Time Augmentation (TTA)
- Experiment with CycleGAN for style transfer

---

## 🛠️ Technologies

- **Framework**: YOLOv8 (Ultralytics)
- **Synthetic Data**: FalconCloud Digital Twin Simulation
- **Language**: Python 3.8+
- **Deep Learning**: PyTorch, CUDA
- **Computer Vision**: OpenCV, PIL

---

## 📚 Resources

- 🏆 [Competition Page](https://www.kaggle.com/competitions/multi-instance-object-detection-challenge)
- 🤖 [YOLOv8 Documentation](https://docs.ultralytics.com/)
- 🌐 [FalconCloud Platform](https://falconcloud.duality.ai)

---

## 📫 Contact

**M. A. Kalam Sharif**  
🎓 B.Sc. Naval Architecture & Marine Engineering - BUET (2025)  
📧 Email: makalamnirjon@gmail.com  
💼 LinkedIn: [Sharif Nirjon](https://www.linkedin.com/in/sharif-nirjon-61977b362/)
🏆 Kaggle: [Nirgenre](https://www.kaggle.com/nirgenre)
💻 GitHub: [@SharifNirjon](https://github.com/SharifNirjon)

---

<div align="center">

### 🎯 Competition Achievement

**36th Place out of 111 Participants**  
**mAP@50: 0.919**  
**Trained on only 89 synthetic images**

---

⭐ **If you found this approach helpful, please star this repo!**

**Proving that smart synthetic data generation beats brute-force approaches! 🚀**

</div>
