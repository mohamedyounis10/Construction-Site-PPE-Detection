"""
Construction Site PPE Detector — Streamlit App
================================================
Deploys the YOLOv8n baseline model trained on the Roboflow
"Construction Site Safety" dataset (10 classes: Hardhat, Mask,
NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone,
Safety Vest, machinery, vehicle).

Run locally:
    streamlit run app.py

Before running, place your trained weights file at:
    weights/best.pt
(or upload a .pt file from the sidebar at runtime).
"""

import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Config
APP_TITLE = "🦺 Construction Site PPE Detector"

# path 
DEFAULT_WEIGHTS_PATH = "ppe_app/weights/ppe_best_model_baseline.pt"
IMG_SIZE = 640

CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask",
    "NO-Safety Vest", "Person", "Safety Cone",
    "Safety Vest", "machinery", "vehicle",
]

# Classes that represent a safety VIOLATION (missing required PPE)
VIOLATION_CLASSES = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}

# Clean, muted per-class colors (RGB) for simple box drawing — kept subtle
# on purpose so the annotated image still reads like a normal photo instead
# of a heavily-stylized "ML demo" image.
CLASS_COLORS_RGB = {
    "Hardhat":        (46, 204, 113),   # green
    "Mask":           (52, 152, 219),   # blue
    "NO-Hardhat":     (231, 76, 60),    # red
    "NO-Mask":        (230, 126, 34),   # orange
    "NO-Safety Vest": (192, 57, 43),    # dark red
    "Person":         (241, 196, 15),   # yellow
    "Safety Cone":    (155, 89, 182),   # purple
    "Safety Vest":    (26, 188, 156),   # teal
    "machinery":      (149, 165, 166),  # grey
    "vehicle":        (52, 73, 94),     # slate
}

# Real numbers pulled from the training notebook's final held-out TEST SET
# evaluation of the deployed model (plain YOLOv8n baseline, 100 epochs,
# re-split 75/15/10 stratified split of the Roboflow CSS dataset).
TEST_SET_OVERALL = {
    "mAP50": 0.756, "mAP50-95": 0.500, "precision": 0.849, "recall": 0.672,
}

TEST_SET_PER_CLASS = pd.DataFrame([
    {"class": "Hardhat",        "precision": 0.898, "recall": 0.650, "mAP50": 0.769, "mAP50-95": 0.472},
    {"class": "Mask",           "precision": 0.901, "recall": 0.792, "mAP50": 0.858, "mAP50-95": 0.626},
    {"class": "NO-Hardhat",     "precision": 0.861, "recall": 0.641, "mAP50": 0.728, "mAP50-95": 0.451},
    {"class": "NO-Mask",        "precision": 0.816, "recall": 0.507, "mAP50": 0.614, "mAP50-95": 0.307},
    {"class": "NO-Safety Vest", "precision": 0.830, "recall": 0.708, "mAP50": 0.782, "mAP50-95": 0.514},
    {"class": "Person",         "precision": 0.887, "recall": 0.752, "mAP50": 0.846, "mAP50-95": 0.616},
    {"class": "Safety Cone",    "precision": 0.786, "recall": 0.512, "mAP50": 0.604, "mAP50-95": 0.282},
    {"class": "Safety Vest",    "precision": 0.861, "recall": 0.688, "mAP50": 0.751, "mAP50-95": 0.480},
    {"class": "machinery",      "precision": 0.871, "recall": 0.850, "mAP50": 0.911, "mAP50-95": 0.728},
    {"class": "vehicle",        "precision": 0.776, "recall": 0.622, "mAP50": 0.696, "mAP50-95": 0.521},
])

PIPELINE_COMPARISON = pd.DataFrame([
    {"stage": "Baseline — YOLOv8n",                        "mAP50": 0.719, "mAP50-95": 0.470},
    {"stage": "Fixed Feature Extraction (frozen backbone)", "mAP50": 0.651, "mAP50-95": 0.406},
    {"stage": "Staged Fine-Tuning (3-stage unfreeze)",       "mAP50": 0.704, "mAP50-95": 0.461},
    {"stage": "Baseline + CBAM attention block",             "mAP50": 0.374, "mAP50-95": 0.216},
    {"stage": "Final combined (CBAM + Staged + Custom Loss)","mAP50": 0.641, "mAP50-95": 0.395},
])

TEST_SET_COMPARISON = pd.DataFrame([
    {"model": "Baseline — YOLOv8n (deployed here)",           "mAP50": 0.756, "mAP50-95": 0.500, "precision": 0.849, "recall": 0.672},
    {"model": "Final combined model (CBAM+Staged+CustomLoss)", "mAP50": 0.669, "mAP50-95": 0.409, "precision": 0.838, "recall": 0.582},
])

st.set_page_config(page_title="PPE Detector", page_icon="🦺", layout="wide")


# Model loading
@st.cache_resource(show_spinner="Loading model weights...")
def load_model(weights_path: str):
    return YOLO(weights_path)


def get_model():
    """Resolve weights from the default path, or let the user point to one."""
    with st.sidebar:
        st.markdown("### ⚙️ Model")
        weights_path = st.text_input(
            "Weights path (.pt)", value=DEFAULT_WEIGHTS_PATH,
            help="Path to best.pt on disk. Change this if your weights live elsewhere.",
        )
        uploaded_weights = st.file_uploader(
            "...or upload a weights file (.pt)", type=["pt"],
            help="Useful for quick local testing without editing the path above.",
        )

    if uploaded_weights is not None:
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "uploaded_weights.pt")
        with open(tmp_path, "wb") as f:
            f.write(uploaded_weights.getbuffer())
        return load_model(tmp_path)

    if os.path.exists(weights_path):
        return load_model(weights_path)

    st.sidebar.error(f"No weights found at `{weights_path}`.")
    return None

# Clean box drawing (kept close to the original photo, thin boxes only)
def draw_clean_boxes(image: np.ndarray, result, is_bgr: bool = False) -> np.ndarray:
    """Draw simple, thin boxes + small labels on a copy of the ORIGINAL
    image — no filled overlays, no saturated ML-demo styling.

    `is_bgr` must match the channel order of `image` (True for raw
    cv2/video frames, False for RGB arrays such as from PIL) so the
    class colors below come out correct either way.
    """
    annotated = image.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return annotated

    boxes = result.boxes.xyxy.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    h, w = annotated.shape[:2]
    thickness = max(1, round(min(h, w) / 160))
    font_scale = max(0.4, min(h, w) / 1000)

    for (x1, y1, x2, y2), cid, conf in zip(boxes, cls_ids, confs):
        name = result.names[int(cid)]
        color_rgb = CLASS_COLORS_RGB.get(name, (255, 255, 255))
        color = color_rgb[::-1] if is_bgr else color_rgb
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        label = f"{name} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_y1 = max(0, y1 - th - baseline - 4)
        label_y2 = max(th + baseline + 4, y1)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + tw + 6, label_y2), color, -1, cv2.LINE_AA)
        text_color = (0, 0, 0) if sum(color_rgb) > 380 else (255, 255, 255)
        cv2.putText(
            annotated, label, (x1 + 3, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA,
        )

    return annotated

# Inference helpers
def summarize_detections(result) -> pd.DataFrame:
    if result.boxes is None or len(result.boxes) == 0:
        return pd.DataFrame(columns=["class", "count"])
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    names = [result.names[i] for i in cls_ids]
    counts = pd.Series(names).value_counts().rename_axis("class").reset_index(name="count")
    return counts


def render_violation_banner(counts_df: pd.DataFrame):
    if counts_df.empty:
        st.info("No objects detected at the current confidence threshold.")
        return
    violations = counts_df[counts_df["class"].isin(VIOLATION_CLASSES)]
    if violations.empty:
        st.success("✅ No PPE violations detected in this frame/image.")
    else:
        items = ", ".join(f"{row['class']} ×{row['count']}" for _, row in violations.iterrows())
        st.error(f"⚠️ Possible PPE violations detected: {items}")

def run_image_inference(model, image: Image.Image, conf: float, iou: float, agnostic_nms: bool):
    image_rgb = np.array(image)
    results = model.predict(
        source=image_rgb, conf=conf, iou=iou, imgsz=IMG_SIZE,
        agnostic_nms=agnostic_nms, verbose=False,
    )
    result = results[0]
    annotated_rgb = draw_clean_boxes(image_rgb, result, is_bgr=False)
    return annotated_rgb, result

def run_video_realtime(
    model,
    input_path: str,
    conf: float,
    iou: float,
    agnostic_nms: bool,
    frame_skip: int = 1,
):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        st.error("Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_delay = 1.0 / fps

    # Placeholder for live video
    video_placeholder = st.empty()

    # Detection statistics
    class_totals = {}

    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Process selected frames
        if frame_idx % frame_skip == 0:

            # YOLO detection
            results = model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                imgsz=IMG_SIZE,
                agnostic_nms=agnostic_nms,
                verbose=False,
            )

            result = results[0]

            # Draw boxes
            annotated = draw_clean_boxes(
                frame,
                result,
                is_bgr=True
            )

            # Update detection statistics
            if result.boxes is not None and len(result.boxes) > 0:
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)

                for cid in cls_ids:
                    name = result.names[int(cid)]
                    class_totals[name] = class_totals.get(name, 0) + 1

        else:
            annotated = frame

        # Convert BGR -> RGB for Streamlit
        annotated_rgb = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

        # Show current frame immediately
        video_placeholder.image(
            annotated_rgb,
            channels="RGB",
            use_container_width=True
        )

        frame_idx += 1

        # Keep approximately the original video FPS
        elapsed = time.time() - start_time
        expected_time = frame_idx * frame_delay

        if expected_time > elapsed:
            time.sleep(expected_time - elapsed)

    cap.release()

    return pd.Series(class_totals).rename_axis(
        "class"
    ).reset_index(
        name="count"
    ) if class_totals else pd.DataFrame(
        columns=["class", "count"]
    )
    
# UI — Sidebar controls shared by the detector tab
def sidebar_inference_controls():
    st.sidebar.markdown("### 🎚️ Detection settings")
    conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou = st.sidebar.slider(
        "IoU threshold (NMS)", 0.10, 0.95, 0.35, 0.05,
        help="Lower = more aggressive suppression of overlapping duplicate "
             "boxes for the SAME class. Lower this if you see two boxes on "
             "the same object.",
    )
    agnostic_nms = st.sidebar.checkbox(
        "Class-agnostic NMS", value=False,
        help="Also suppresses overlapping boxes ACROSS different classes, "
             "keeping only the highest-confidence one. Turn on only if you "
             "see boxes from different classes stacked on the same object "
             "— it can hide legitimate cases like a Person box overlapping "
             "a NO-Safety Vest box, which is normal.",
    )
    return conf, iou, agnostic_nms

# Tab 1 — Live detection
def detection_tab(model):
    conf, iou, agnostic_nms = sidebar_inference_controls()

    mode = st.radio("Input type", ["Image", "Video"], horizontal=True)

    if mode == "Image":
        uploaded = st.file_uploader(
            "Upload a construction-site photo",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            with st.spinner("Running detection..."):
                annotated, result = run_image_inference(model, image, conf, iou, agnostic_nms)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", use_container_width=True)
            with col2:
                st.image(annotated, caption="Detected", use_container_width=True)

            counts_df = summarize_detections(result)
            render_violation_banner(counts_df)
            if not counts_df.empty:
                st.markdown("**Detections in this image**")
                st.dataframe(counts_df, use_container_width=True, hide_index=True)

    else:  # Video
        uploaded = st.file_uploader(
        "Upload a construction-site video clip",
        type=["mp4", "mov", "avi", "mkv"]
         )

        frame_skip = st.number_input(
          "Process every Nth frame",
          min_value=1,
          max_value=10,
          value=1,
          help="1 = process every frame for smoother real-time detection."
        )

        if uploaded is not None:

        # Save uploaded video temporarily
           tmp_dir = tempfile.mkdtemp()

           input_path = os.path.join(
            tmp_dir,
            uploaded.name
           )
           
           with open(input_path, "wb") as f:
               f.write(uploaded.getbuffer())

           st.info("🎥 Video ready — click Start to begin real-time detection.")

           if st.button(
            "▶️ Start Real-Time Detection",
             type="primary"
           ):

               st.markdown("### 🔴 Live Detection")

               totals_df = run_video_realtime(
                model,
                input_path,
                conf,
                iou,
                agnostic_nms,
                int(frame_skip)
               )

               st.success("✅ Video finished.")

               if totals_df is not None and not totals_df.empty:

                  render_violation_banner(totals_df)

                  st.markdown(
                    "**Total detections during video**"
                  )

                  st.dataframe(
                    totals_df,
                    use_container_width=True,
                    hide_index=True
                  )

# Tab 2 — About this project
def about_tab():
    st.markdown("## 🏗️ About This Project")
    st.markdown(
        """
This app detects **personal protective equipment (PPE)** on construction sites —
hard hats, safety vests, masks — and flags when a worker appears to be
**missing** required gear, from a photo or a video clip.
        """
    )

    st.markdown("### In simple terms")
    st.markdown(
        """
The model looks at an image or a video frame and draws a box around every
person, piece of safety gear, or vehicle/machine it recognizes — 10 categories
in total, including three **"missing gear" categories** (`NO-Hardhat`,
`NO-Mask`, `NO-Safety Vest`) that act as automatic violation alerts. The
underlying detector is **YOLOv8n**, a small, fast object-detection network
(about 3 million parameters, ~6 MB), which is why it can run in real time on
a normal CPU and even faster on a GPU.
        """
    )

    with st.expander("📦 Dataset"):
        st.markdown(
            """
- **Source:** Roboflow's *Construction Site Safety* dataset (YOLO format,
  CC BY 4.0), mirrored on Kaggle by `snehilsanyal`.
- **Original split:** 2,605 train / 114 valid / 82 test images (2,801 total).
- **Re-split used for training:** the project pooled all images and
  re-divided them **75 / 15 / 10** (stratified by each image's rarest
  class) to get a bigger, more reliable validation and test set —
  2,097 / 414 / 290 images.
- **Classes (10):** Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest,
  Person, Safety Cone, Safety Vest, machinery, vehicle.
- **Class imbalance:** `Person` (9,532 instances) and `machinery` (5,247)
  dominate, while `Mask` (1,651) and `vehicle` (1,545) are comparatively
  rare — a real challenge for training and a reason accuracy on those
  classes is lower (see the per-class table below).
            """
        )

    st.markdown("### The pipeline — what was actually tried")
    st.markdown(
        """
This wasn't a single training run. The project systematically compared
several approaches on the same data and split, in this order:
        """
    )
    st.markdown(
        """
1. **Baseline model comparison** — trained plain `YOLOv8n` and `YOLO11n`
   from their COCO checkpoints on the PPE data; **YOLOv8n came out ahead**
   and was carried forward as the base model for everything below.
2. **Transfer-learning strategy audit** — compared *Fixed Feature
   Extraction* (freeze the backbone, only train the head) against a
   *3-stage progressive fine-tuning* schedule (unfreeze more of the
   backbone in each stage, with a shrinking learning rate). Staged
   fine-tuning won.
3. **Custom architecture extension** — inserted a **CBAM** (Convolutional
   Block Attention Module — channel + spatial attention) layer right
   after the backbone's SPPF block, to see if attention would help the
   model focus on small/occluded PPE items.
4. **Custom loss** — replaced the default classification loss with a
   **class-balanced BCE loss**, weighting each class by inverse frequency
   so rare classes (`Mask`, `vehicle`) aren't drowned out by `Person`.
5. **Final combined model** — CBAM + staged fine-tuning + the
   class-balanced loss, trained together as the "proposed" model.
        """
    )

    st.markdown("### Validation-set results across the pipeline")
    st.dataframe(
        PIPELINE_COMPARISON.style.format({"mAP50": "{:.3f}", "mAP50-95": "{:.3f}"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("### ⚠️ The honest finding")
    st.warning(
        """
**None of the added complexity beat the plain baseline.** On the held-out
test set, the plain, well-tuned YOLOv8n baseline scored **mAP50 = 0.756**,
while the "final" CBAM + staged-fine-tuning + custom-loss model scored
**mAP50 = 0.669** — noticeably lower, despite being the most sophisticated
version. That's why **this app deploys the plain baseline**, not the
fancier model.
        """
    )
    st.dataframe(
        TEST_SET_COMPARISON.style.format(
            {"mAP50": "{:.3f}", "mAP50-95": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}"}
        ),
        use_container_width=True, hide_index=True,
    )

    with st.expander("🔍 For readers with an ML background — why did the fancy version lose?"):
        st.markdown(
            """
A couple of likely, non-mutually-exclusive causes surfaced while building
this:

- **Partial weight transfer after inserting CBAM.** Adding a new layer
  into the architecture shifts every later layer's index. When the
  pretrained COCO weights were transferred into the CBAM-augmented graph,
  only **162 of 358** state-dict tensors matched and copied over — the
  rest of the detection head (and CBAM itself) started from scratch. That
  alone explains most of the collapse in the isolated CBAM-only ablation
  (mAP50 dropped from 0.719 → 0.374, a 54% relative hit on mAP50-95).
- **Not enough fine-tuning budget to recover.** The staged schedule that
  worked well for the *plain* backbone (10/10/15 epochs per stage) wasn't
  enough to let a partially-reinitialized head re-converge once CBAM and
  the custom loss were layered on top — each stage was tuned for the
  easier fixed-backbone case, not for recovering from a broken weight
  transfer.
- **Small dataset, small model.** With ~2,800 images and a 3M-parameter
  network, there's limited headroom for an extra attention module to pay
  for itself — the added ~8.3K CBAM parameters (a 0.27% increase) simply
  don't have much signal to learn from once the rest of the head is
  destabilized.

This is a legitimate, useful result, not a wasted effort: it shows that a
carefully-tuned simple baseline can beat a "smarter" architecture when the
integration isn't done carefully — and it points exactly at where a next
iteration should focus (fix the weight-transfer mapping, or give the
CBAM branch its own warm-up stage before combining it with staged
fine-tuning and the custom loss).
            """
        )

    st.markdown("### 📋 Deployed model card")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Architecture", "YOLOv8n")
    c2.metric("Parameters", "~3.0M")
    c3.metric("File size", "~6 MB")
    c4.metric("Input size", "640×640")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test mAP50", f"{TEST_SET_OVERALL['mAP50']:.3f}")
    c2.metric("Test mAP50-95", f"{TEST_SET_OVERALL['mAP50-95']:.3f}")
    c3.metric("Test precision", f"{TEST_SET_OVERALL['precision']:.3f}")
    c4.metric("Test recall", f"{TEST_SET_OVERALL['recall']:.3f}")

    st.caption(
        "GPU/CPU latency was benchmarked on the near-identical CBAM sibling model "
        "(3.02M params vs. 3.01M here): ~7 ms/frame on a Tesla T4 GPU, ~88 ms/frame "
        "on CPU — a reasonable proxy since the two models are almost the same size."
    )

    st.markdown("### Per-class performance (held-out test set)")
    st.dataframe(
        TEST_SET_PER_CLASS.style.format(
            {"precision": "{:.3f}", "recall": "{:.3f}", "mAP50": "{:.3f}", "mAP50-95": "{:.3f}"}
        ).background_gradient(subset=["mAP50"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "`NO-Mask` and `Safety Cone` are the weakest classes (small objects, and "
        "`NO-Mask` is visually subtle) — worth extra scrutiny before trusting an "
        "alert on those two classes in a real deployment."
    )

    with st.expander("🚧 Limitations & responsible-use notes"):
        st.markdown(
            """
- Trained on **~2,800 images** from one public dataset — performance on a
  specific real site (different camera angle, lighting, PPE colors/styles)
  is **not guaranteed** and should be validated before operational use.
- This is a **detector, not a certified safety system**. It should support
  human safety officers, not replace them — false negatives (missed
  violations) and false positives (false alarms) will happen, as the
  per-class table above shows.
- Recall on `NO-Mask` and `Safety Cone` is the lowest in the model — treat
  low-confidence detections in those classes with extra caution.
- The model was evaluated on held-out images/video frames from the *same*
  data distribution as training; it has not been stress-tested against
  adversarial conditions (heavy rain, night footage, extreme occlusion).
            """
        )

    st.markdown("### 📚 Source & citation")
    st.markdown(
        """
- Dataset: [Construction Site Safety Image Dataset — Roboflow, on Kaggle]
  (https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)
  · originally published on [Roboflow Universe]
  (https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety),
  licensed CC BY 4.0.
- Detector: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).
        """
    )

# Main
def main():
    st.title(APP_TITLE)
    model = get_model()

    tab1, tab2 = st.tabs(["🔎 Detect PPE", "📖 About this project"])

    with tab1:
        if model is None:
            st.warning(
                f"Place your trained weights at `{DEFAULT_WEIGHTS_PATH}` (or upload "
                "one from the sidebar) to start detecting."
            )
        else:
            detection_tab(model)

    with tab2:
        about_tab()

if __name__ == "__main__":
    main()
