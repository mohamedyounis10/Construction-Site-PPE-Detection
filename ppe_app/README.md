# 🦺 Construction Site PPE Detector — Streamlit App

A ready-to-deploy Streamlit app around the trained YOLOv8n PPE model:
upload an image or a video and it draws boxes for hard hats, safety
vests, masks, people, machinery/vehicles, and flags missing PPE
(`NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`) as violations. It also
ships a built-in **"About this project"** tab that explains the whole
pipeline and its results (see `app.py` for the content).

## 1. Get the weights file

The notebook's own final step saves the best-performing model to:

```
/kaggle/working/ppe_best_model_baseline.pt
```

Download that file from your Kaggle output, rename it to `best.pt`,
and place it here:

```
ppe_app/
└── weights/
    └── best.pt   ← put it here
```

(Why the "baseline" file and not the fancier CBAM one? Because on the
held-out test set the plain baseline actually scored higher — see the
About tab in the app for the full explanation.)

If you'd rather not touch the file system, you can also just upload a
`.pt` file from the app's sidebar at runtime — handy for quick local
testing, but for a real deployment (Streamlit Community Cloud, Docker,
etc.) committing `weights/best.pt` to the project is the simplest
approach.

## 2. Install & run locally

```bash
cd ppe_app
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 3. Deploy

**Streamlit Community Cloud** (free, easiest):
1. Push this folder to a GitHub repo (include `weights/best.pt` — use
   [Git LFS](https://git-lfs.com/) if your Git host limits file size,
   though at ~6 MB this file is small enough for a normal commit).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the
   repo, and point it at `app.py`.

**Docker** (any cloud VM / container platform):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

## 4. Where to get test images/videos

The dataset itself (same one the model was trained on — perfect for a
fair first test since the model has already "seen" similar imagery):

- **Kaggle mirror (images + labels, easy zip download):**
  https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow
  — grab a few files from the `css-data/test/images/` folder for
  images the model was *not* trained on.
- **Roboflow Universe (browse/preview in-browser, export subsets, or
  hit their inference API directly):**
  https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
- **Original results/report repo (also links a sample notebook + a
  few example predictions):**
  https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection

For material the model has genuinely never seen (better stress test
for how it'll behave on your own site footage):

- **Free stock video** — search "construction site", "hard hat",
  "safety vest" on [Pexels Videos](https://www.pexels.com/videos/) or
  [Pixabay Videos](https://pixabay.com/videos/) (free, no attribution
  required, good variety of angles/lighting).
- **Your own phone footage** — a quick clip of yourself (or a
  colleague) with/without a hard hat or vest is the most realistic
  test of how the model behaves outside its training distribution.
- **Google/Bing image search** for "construction worker PPE" style
  queries, filtered to reusable/Creative-Commons images if you plan to
  reuse them beyond a private demo.

## Files

```
ppe_app/
├── app.py              # Streamlit app (detection + about tab)
├── requirements.txt
├── README.md
└── weights/
    └── best.pt          # ← you provide this (see step 1)
```
