# GhostEye
Fine-tuned YOLOv8 model for detecting military assets in aerial imagery using the DOTA dataset — trained on Google Colab with OpenCV visualization and ONNX export.
# 🎯 Military Vehicle Detection using YOLOv8 + OpenCV

A deep learning project for detecting military assets in aerial/satellite imagery using **YOLOv8** transfer learning on the **DOTA dataset**, with full OpenCV visualization and ONNX export support.

---

## 📌 Project Overview

This project fine-tunes a YOLOv8 model to detect military vehicles and objects (planes, ships, tanks, helicopters, etc.) from overhead imagery. It covers the full ML pipeline — from data loading and training to inference, benchmarking, and deployment export.

| Stage | Details |
|---|---|
| **Dataset** | DOTA v1 (via Roboflow) |
| **Model** | YOLOv8n (pretrained on COCO) |
| **Framework** | Ultralytics + OpenCV |
| **Runtime** | Google Colab (T4 GPU) |
| **Export** | ONNX (opset 12) |

---

## 🗂️ Notebook Structure

| Step | Description |
|---|---|
| Step 1 | Install dependencies |
| Step 2 | Download DOTA dataset via Roboflow |
| Step 3 | Verify dataset splits (train / val / test) |
| Step 4 | Visualize ground truth labels |
| Step 5 | Train YOLOv8n for 30 epochs |
| Step 6 | Evaluate model (mAP, Precision, Recall) |
| Step 7 | Run inference with OpenCV bounding boxes |
| Step 8 | Benchmark inference speed (FPS / latency) |
| Step 9 | Export model to ONNX |

---

## 🚀 Getting Started

### Run on Google Colab (Recommended)

1. Open the notebook in Colab
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Run all cells top to bottom (**Runtime → Run all**)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soumyathamke/GhostEye/blob/main/military_vehicle_detection_yolov8_FIXED.ipynb)

> ⚠️ Replace `YOUR_USERNAME` with your GitHub username after uploading.

### Local Setup

```bash
git clone https://github.com/soumyathamke/GhostEye.git
cd GhostEye
pip install -r requirements.txt
jupyter notebook military_vehicle_detection_yolov8_FIXED.ipynb
```

---

## 📦 Requirements

```
ultralytics>=8.3.0
roboflow
opencv-python-headless
matplotlib
Pillow
torch
PyYAML
numpy
```

> A full pinned list is in `requirements.txt`.

---

## 🧠 Model Training Config

```python
model = YOLO("yolov8n.pt")
model.train(
    data="data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    optimizer="AdamW",
    lr0=0.001,
    augment=True,
    patience=10,
)
```

Training takes approximately **~20 minutes on a T4 GPU**.

---

## 📊 Evaluation Metrics

After training, the model is evaluated on:

- **mAP@0.5** — Mean Average Precision at IoU 0.5
- **mAP@0.5:0.95** — COCO-style mAP
- **Precision** and **Recall**
- Per-class AP breakdown with visual bar display

---

## ⚡ Inference & Benchmarking

- Custom OpenCV visualization with colored bounding boxes per class
- Detection count overlay on each frame
- Benchmarked on 50 validation images: reports **avg latency (ms)** and **FPS**

---

## 📤 ONNX Export

The trained model is exported to ONNX format for deployment:

```python
model.export(format="onnx", imgsz=640, opset=12, simplify=True)
```

---

## 🗃️ Dataset

This project uses the **DOTA (Detection in Optical Remote Sensing Images)** dataset hosted on Roboflow.

- **Classes include:** plane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, large vehicle, small vehicle, helicopter, roundabout, soccer ball field, swimming pool
- Dataset is automatically downloaded in Step 2 via the Roboflow API

---

## 📁 Project Structure

```
military-vehicle-detection/
├── military_vehicle_detection_yolov8_FIXED.ipynb   # Main notebook
├── requirements.txt                                  # Python dependencies
├── .gitignore                                        # Files to exclude from git
└── README.md                                         # This file
```

---

## 🛠️ Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com/)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [Google Colab](https://colab.research.google.com/)

---

## 📄 License

This project is for educational and research purposes. Dataset usage is subject to [DOTA dataset license terms](https://captain-whu.github.io/DOTA/).
