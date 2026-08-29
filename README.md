<div align="center">
  <h1>Construction Site PPE Detection 🦺🏗️🔍</h1>
  <p>Final graduation project for the <strong>Axis Training Program</strong> 🎓 — an end-to-end computer vision pipeline for detecting personal protective equipment on construction sites (data prep 🧹 → YOLOv8 training 🧠 → evaluation 📈 → Streamlit deployment 🚀) using Python and Ultralytics.</p>

  <p>
    <a href="#overview">Overview 🧾</a> •
    <a href="#project-structure">Project Structure 🗂️</a> •
    <a href="#dataset">Dataset 🧩</a> •
    <a href="#results">Results 📊</a> •
    <a href="#streamlit-app">Streamlit App 🖥️</a>
  </p>

  <p>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.x-blue" />
    <img alt="YOLOv8" src="https://img.shields.io/badge/YOLOv8-Object_Detection-00FFFF" />
    <img alt="Ultralytics" src="https://img.shields.io/badge/Ultralytics-8.x-111111" />
    <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Web_App-FF4B4B" />
    <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8" />
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C" />
  </p>
</div>

---

## Table of Contents 🧭

- [Overview 🧾](#overview)
- [Project Structure 🗂️](#project-structure)
- [Dataset 🧩](#dataset)
- [Results 📊](#results)
- [Streamlit App 🖥️](#streamlit-app)
- [How to Run ▶️](#how-to-run)
- [Author ✍️](#author-️)

---

<a id="overview"></a>
## Overview 🧾

This repository is the **final project** submitted as part of the **Axis Training Program** in computer vision & deep learning. It focuses on **Construction Site PPE (Personal Protective Equipment) Detection**, building a real-time safety monitoring system that identifies workers, safety gear, and PPE violations from images and video through:

- **Data Preparation 🧹**: Stratified re-splitting of the Roboflow Construction Site Safety dataset (75 / 15 / 10) for reliable validation and test evaluation.
- **Model Training 🧠**: Systematic comparison of transfer-learning strategies — baseline YOLOv8n, frozen-backbone extraction, staged fine-tuning, CBAM attention, and class-balanced loss.
- **Evaluation 📈**: Held-out test-set metrics (mAP50, mAP50-95, precision, recall) with per-class breakdown across 10 safety categories.
- **Deployment 🚀**: Interactive **Streamlit** web app for image/video inference, violation alerts, and a built-in project walkthrough.

**Key finding:** The plain, well-tuned **YOLOv8n baseline** outperformed all more complex variants on the test set — so that is the model shipped in this repo.

---

<a id="project-structure"></a>
## Project Structure 🗂️

- [`ppe_app/` 🖥️](#streamlit-app) — deployable Streamlit application
  - `app.py` — detection UI + "About this project" tab
  - `requirements.txt` — Python dependencies
  - `weights/ppe_best_model_baseline.pt` — trained YOLOv8n weights (~6 MB)
- [`Try - Images & Videos/` 📸](#how-to-run) — sample images & clips for quick testing

Tree 🌳:

```text
Project - Final/
├─ ppe_app/
│  ├─ app.py
│  ├─ requirements.txt
│  ├─ README.md
│  └─ weights/
│     └─ ppe_best_model_baseline.pt
└─ Try - Images & Videos/
   ├─ *.jpg / *.jpeg          (sample construction-site photos)
   └─ *.mp4                    (sample construction-site videos)
```

---

<a id="dataset"></a>
## Dataset 🧩

The model is trained on the **Construction Site Safety** dataset from Roboflow (YOLO format, CC BY 4.0), mirrored on Kaggle by `snehilsanyal`.

| Split | Images |
|---|---:|
| **Original (Roboflow)** | 2,801 total (2,605 train / 114 valid / 82 test) |
| **Re-split used (75/15/10, stratified)** | 2,801 total (2,097 train / 414 valid / 290 test) |

**Classes (10) 🎯:**

| Category | Classes |
|---|---|
| **PPE worn** | `Hardhat`, `Mask`, `Safety Vest` |
| **PPE violations** | `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest` |
| **Scene context** | `Person`, `Safety Cone`, `machinery`, `vehicle` |

**Class imbalance note:** `Person` (9,532 instances) and `machinery` (5,247) dominate the dataset, while `Mask` (1,651) and `vehicle` (1,545) are comparatively rare — which affects per-class recall.

**Dataset links:**
- [Kaggle mirror](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)
- [Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)

---

<a id="results"></a>
## Results 📊

### Deployed Model — Test Set KPIs 🧪

| Metric | Value |
|---|---:|
| **mAP50** | **0.756** |
| **mAP50-95** | **0.500** |
| **Precision** | **0.849** |
| **Recall** | **0.672** |
| **Architecture** | YOLOv8n (~3.0M params, ~6 MB) |
| **Input size** | 640 × 640 |

### Pipeline Comparison (Validation Set) 🔬

| Stage | mAP50 | mAP50-95 |
|---|---:|---:|
| **Baseline — YOLOv8n** | **0.719** | **0.470** |
| Fixed Feature Extraction (frozen backbone) | 0.651 | 0.406 |
| Staged Fine-Tuning (3-stage unfreeze) | 0.704 | 0.461 |
| Baseline + CBAM attention block | 0.374 | 0.216 |
| Final combined (CBAM + Staged + Custom Loss) | 0.641 | 0.395 |

### Test Set — Baseline vs. Final Combined Model ⚖️

| Model | mAP50 | mAP50-95 | Precision | Recall |
|---|---:|---:|---:|---:|
| **Baseline — YOLOv8n (deployed)** | **0.756** | **0.500** | **0.849** | **0.672** |
| Final combined (CBAM+Staged+CustomLoss) | 0.669 | 0.409 | 0.838 | 0.582 |

### Per-Class Performance (Test Set) 📋

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| **Hardhat** | 0.898 | 0.650 | 0.769 | 0.472 |
| **Mask** | 0.901 | 0.792 | 0.858 | 0.626 |
| **NO-Hardhat** | 0.861 | 0.641 | 0.728 | 0.451 |
| **NO-Mask** | 0.816 | 0.507 | 0.614 | 0.307 |
| **NO-Safety Vest** | 0.830 | 0.708 | 0.782 | 0.514 |
| **Person** | 0.887 | 0.752 | 0.846 | 0.616 |
| **Safety Cone** | 0.786 | 0.512 | 0.604 | 0.282 |
| **Safety Vest** | 0.861 | 0.688 | 0.751 | 0.480 |
| **machinery** | 0.871 | 0.850 | 0.911 | 0.728 |
| **vehicle** | 0.776 | 0.622 | 0.696 | 0.521 |

### Insights 💡

- **Baseline wins:** Despite testing CBAM attention, staged fine-tuning, and class-balanced loss, the plain YOLOv8n baseline achieved the highest test mAP50 (0.756 vs. 0.669).
- **Strong on machinery:** `machinery` reached the highest mAP50 (0.911) and recall (0.850) among all classes.
- **Weak spots:** `NO-Mask` and `Safety Cone` are the hardest classes — small size and subtle visual cues drive lower recall.
- **Real-time capable:** ~7 ms/frame on GPU (Tesla T4) and ~88 ms/frame on CPU — suitable for live video monitoring.

---

<a id="streamlit-app"></a>
## Streamlit App 🖥️

🚀 **Live Demo:** The application is now hosted and live! You can try it out directly here: **[https://construction-site-ppe-detection-dn9yzcealhudmnvrggnf2u.streamlit.app/]**

An interactive **Streamlit web application** (`ppe_app/app.py`) provides:

- **Image detection** — upload a photo, see original vs. annotated output side-by-side.
- **Video detection** — real-time frame-by-frame inference with configurable frame skip.
- **Violation alerts** — automatic flags for `NO-Hardhat`, `NO-Mask`, and `NO-Safety Vest`.
- **Tunable settings** — confidence threshold, IoU (NMS), and class-agnostic NMS from the sidebar.
- **About tab** — full project narrative, dataset details, pipeline comparison, and model card.

The trained weights are included at `ppe_app/weights/ppe_best_model_baseline.pt`.

<img width="1892" height="897" alt="Screenshot 2026-08-29 142538" src="https://github.com/user-attachments/assets/b3dfea03-dffd-4770-9ff6-d2ada9ff0515" />
<img width="1892" height="882" alt="Screenshot 2026-08-29 142518" src="https://github.com/user-attachments/assets/5f0e714f-0618-4af8-8780-1797861a5eae" />

---

<a id="how-to-run"></a>
## How to Run ▶️

### 1) Setup Environment 🧪

```bash
cd ppe_app
python -m venv .venv
.\.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
```

### 2) Install Dependencies 📦

```bash
pip install -r requirements.txt
```

### 3) Launch the App 🚀

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser, then upload an image or video from `Try - Images & Videos/` (or your own construction-site footage).

---

## Author ✍️

- **Name**: Mohamed Younis
- **Program**: Axis Training Program 🎓 — Computer Vision & Deep Learning track

---
