# Real-Time Anomaly Detection and Crowd Monitoring

An interactive Streamlit dashboard for two surveillance analytics use cases:

1. **Avenue Dataset** — Frame-level anomaly classification (normal / unusual action / abnormal object) using ResNet-34 + YOLOv8.
2. **ShanghaiTech Dataset** — Crowd density estimation and overcrowding alerts using CSRNet with YOLOv8 fallback.

---

## Motivation and Background

Traditional surveillance systems rely on human operators to monitor camera feeds continuously — a process prone to fatigue, distraction, and delayed response. This project automates anomaly detection and crowd monitoring using deep learning, enabling scalable, real-time analysis of surveillance footage without manual intervention.

Specifically, the system targets:
- **Overcrowding** in public spaces (stations, plazas, venues)
- **Behavioral anomalies** such as running, loitering, or carrying prohibited objects
=======
An AI-powered surveillance system combining **YOLOv8**, **ResNet18**, and **CSRNet** to detect anomalous behavior and monitor crowd density in real time. Built as a UMBC DATA606 Capstone Project and deployed as an interactive Streamlit application.

🔗 **[Live Demo — Launch the App](https://anomalydetectioncapstone-aagua27sjxluyvjretnrul.streamlit.app)**

---

## Performance Highlights

| Dataset | Model | Metric | Score |
|---|---|---|---|
| ShanghaiTech Part A | CSRNet | MAE | 207.91 |
| ShanghaiTech Part A | CSRNet | RMSE | 317.36 |
| ShanghaiTech Part B | CSRNet | MAE | **31.18** |
| ShanghaiTech Part B | CSRNet | RMSE | 66.59 |
| Avenue Dataset | YOLOv8 + ResNet18 | Loss (original) | 92.17 |
| Avenue Dataset | YOLOv8 + ResNet18 | Loss (improved) | **68.76** |

---

## What It Does

The system addresses two core surveillance challenges:

1. **Anomaly Detection (Avenue Dataset)** — Classifies unusual behavior in pedestrian surveillance footage using YOLOv8 for object detection and a fine-tuned ResNet18 classifier. Detects running, throwing, abnormal objects (bikes, bags, carts), and unusual movement directions.

2. **Crowd Density Estimation (ShanghaiTech Dataset)** — Generates density maps and estimates head counts using CSRNet with a YOLOv8 fallback for robust detection. Triggers overcrowding alerts when count exceeds 20.

---

## Motivation

Traditional surveillance depends on human operators who are prone to fatigue, distraction, and delayed response. As environments grow more complex, automated AI-driven detection becomes essential for scalable, accurate, real-time monitoring. This project integrates deep learning models into a production-ready Streamlit interface designed for practical deployment.
>>>>>>> 94a932239310e22c27eb771bc92b03beecdf9483

---

## Project Structure

```
AnomalyDetection_Capstone/
│
├── app.py                               # Streamlit dashboard (thin UI layer)
│
├── src/                                 # Application logic package
│   ├── config.py                        # All tunable parameters + env-var overrides
│   ├── models.py                        # CSRNet and AnomalyClassifier definitions
│   ├── inference.py                     # Model loading and inference functions
│   └── utils.py                         # Logging, temp-file management, validation
│
├── notebooks/
│   ├── Avenue_Anomaly_Detection.ipynb   # YOLOv8 + ResNet-34 training pipeline
│   └── ShanghaiTech_Overcrowding.ipynb  # CSRNet training pipeline
│
├── assets/
│   ├── diagram-avenue.png               # Avenue pipeline architecture diagram
│   └── diagram-shanghaitech.png         # CSRNet pipeline architecture diagram
│
├── references/                          # Supporting research papers (PDF)
│
├── requirements.txt                     # Pinned Python dependencies
├── .env.example                         # Environment variable template
=======
root/
├── app.py                               # Streamlit dashboard combining both modules
├── notebooks/
│   ├── Avenue_Anomaly_Detection.ipynb   # YOLOv8 + ResNet18 for anomaly detection
│   └── ShanghaiTech_Overcrowding.ipynb  # CSRNet-based crowd density estimation
├── assets/
│   ├── avenue_pipeline.png              # Anomaly detection architecture diagram
│   └── csrnet_pipeline.png              # Crowd monitoring architecture diagram
├── references/                          # Supporting research papers (PDF)
├── requirements.txt
└── README.md
```

---

## Architecture

### 1. Clone the repository
=======
### Avenue Dataset — Anomaly Detection (YOLOv8 + ResNet18)

![Avenue Pipeline](assets/diagram-avenue.png)

- **Object Detection:** YOLOv8 detects people and objects per frame
- **Classification:** Fine-tuned ResNet18 classifies each frame as Normal / Unusual Action / Abnormal Object
- **Input:** Video frames extracted from `.avi` files
- **Output:** Annotated video with bounding boxes and class labels
- **Labels:** Normal, Unusual Action, Abnormal Object

### ShanghaiTech — Crowd Monitoring (CSRNet)

![CSRNet Pipeline](assets/diagram-shanghaitech.png)

- **Model:** CSRNet with VGG16 frontend and dilated convolutional backend
- **Input:** `.jpg` images with annotated `.mat` ground truth (head coordinates)
- **Output:** Density map + total head count estimate
- **Alert:** Overcrowding triggered when count exceeds 20

---

## Datasets

### Avenue Dataset
- 16 training videos (normal behavior only), 21 test videos (with anomalies)
- ~15,328 test frames at 640x360 resolution, ~25 FPS
- Static camera on a single pedestrian avenue
- Anomalies: running, throwing objects, abnormal directions, bikes, bags, carts

### ShanghaiTech Dataset
- **Part A** (dense crowds): 300 train / 182 test annotated images
- **Part B** (sparse crowds): 400 train / 316 test annotated images
- Ground truth: `.mat` files with head coordinates converted to Gaussian density maps

---

## Setup

```bash
git clone https://github.com/stutiupadhyay03/AnomalyDetection_Capstone.git
cd AnomalyDetection_Capstone
```

### 2. Create and activate a virtual environment

=======
```bash
# Optional: create a virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### 3. Install dependencies

=======
source venv/bin/activate  # Windows: venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

### 4. Download model files

Place the following files in the `notebooks/` directory (default) or set the `MODEL_DIR` environment variable to override:

| File | Description |
|------|-------------|
| `yolov8m.pt` | YOLOv8 medium — from [Ultralytics](https://github.com/ultralytics/ultralytics) |
| `csrnet_shanghai.pt` | Trained CSRNet state-dict |
| `final_op_avenue_model.pt` | Trained ResNet-34 state-dict |

To use custom paths, copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

### 5. Run the app
=======
Download the required model files and place them in the root directory:
- `yolov8m.pt` — from [Ultralytics](https://github.com/ultralytics/ultralytics)
- `csrnet_shanghai.pt` — pretrained CSRNet model
- `final_op_avenue_model.pt` — fine-tuned ResNet18 classifier for Avenue

```bash
streamlit run app.py
# Open: http://localhost:8501
```

### 6. Open in browser

Navigate to `http://localhost:8501`

---

## Datasets

### ShanghaiTech

Designed for crowd counting and density estimation, adapted here for overcrowding alerts.

| Split | Part A (dense) | Part B (sparse) |
|-------|---------------|-----------------|
| Train | 300 images | 400 images |
| Test  | 182 images | 316 images |

Annotations are head coordinates in `.mat` files, converted to Gaussian density maps (σ = 15).

### Avenue

Designed for video anomaly detection in surveillance scenarios.

- 16 training videos (normal behavior only)
- 21 test videos (with labeled anomalies)
- Resolution: 640×360 @ ~25 FPS, ~15,328 test frames

Anomaly types: running, throwing objects, loitering, carrying bicycles or bags.

---

## Model Highlights

### ShanghaiTech — CSRNet Crowd Counting

| Property | Value |
|----------|-------|
| Architecture | CSRNet (VGG-16 frontend + dilated conv backend) |
| Input | Images, any resolution |
| Output | Single-channel density map; sum = crowd count |
| Training | 100 epochs, Adam (lr=1e-5), MSELoss, combined Part A+B |
| Part A metrics | MAE: 207.91 · RMSE: 317.36 |
| Part B metrics | MAE: 31.18 · RMSE: 66.59 |

**Alert levels:**

| Count | Label |
|-------|-------|
| ≤ 5 | Normal |
| ≤ 10 | Can lead to overcrowding |
| ≤ 20 | Possible overcrowding |
| > 20 | Overcrowding |

### Avenue — ResNet-34 Anomaly Classifier

| Property | Value |
|----------|-------|
| Architecture | ResNet-34 + Dropout(0.5) + Linear(512→3) |
| Object detector | YOLOv8m (for rule-based label generation) |
| Input | Video frames at 224×224 |
| Output | 3-class probabilities (normal / unusual action / abnormal object) |
| Training | Adam (lr=1e-4, weight_decay=1e-5), CrossEntropyLoss, grouped train/val split |

---

## Architecture Diagrams

### ShanghaiTech Crowd Monitoring (CSRNet)

![CSRNet Pipeline](assets/diagram-shanghaitech.png)

### Avenue Anomaly Detection (ResNet-34 + YOLOv8)

![Avenue Pipeline](assets/diagram-avenue.png)

---

## Live Demo

Try the hosted Streamlit dashboard:

**[Launch the app](https://anomalydetectioncapstone-aagua27sjxluyvjretnrul.streamlit.app)**

Features:
- Upload crowd images or videos → receive density maps and overcrowding alerts
- Upload surveillance footage → download annotated video with frame-level labels

---

## Future Directions

1. **Live stream support** — Integrate RTSP/webcam feeds via OpenCV for active monitoring.
2. **Temporal modeling** — Add LSTM or Transformer layers after ResNet-34 to capture motion context across frames.
3. **Extended anomaly classes** — Fighting, climbing, tailgating, unauthorized gatherings.
4. **Alert notifications** — Email/SMS alerts via Twilio or SendGrid when thresholds are exceeded.
5. **Edge deployment** — Optimize with TensorRT or ONNX for Jetson-class edge devices.
=======
---

## Future Directions

- **Live stream integration** — CCTV/webcam feeds via OpenCV/RTSP
- **Temporal modeling** — LSTM/GRU/Transformer after ResNet for motion context across frames
- **Expanded anomaly classes** — fighting, climbing, tailgating, unauthorized gatherings
- **Alert system** — email/SMS/dashboard notifications via Twilio or SendGrid
- **Edge deployment** — Jetson devices for real-time inference; AWS/Azure/GCP for scalable cloud deployment
>>>>>>> 94a932239310e22c27eb771bc92b03beecdf9483

---

## Developed By

<<<<<<< HEAD
- **Stuti Upadhyay** — UMBC | DATA 606 Capstone
- **Lakshmi Tejaswini Chandra Pampana** — UMBC | DATA 606 Capstone
=======
- **Stuti Upadhyay** — UMBC | DATA606 Capstone | [github.com/stutiupadhyay03](https://github.com/stutiupadhyay03)
- **Lakshmi Tejaswini Chandra Pampana** — UMBC | DATA606 Capstone

---

## Stack

`PyTorch` · `YOLOv8 (Ultralytics)` · `ResNet18` · `CSRNet` · `OpenCV` · `Streamlit` · `Python`
---

## References

<<<<<<< HEAD
1. Choudhry, N., Abawajy, J., Huda, S., & Rao, I. *A Comprehensive Survey of Machine Learning Methods for Surveillance Videos Anomaly Detection.* Deakin University.

2. Abbas, Z. K., & Al-Ani, A. A. *Anomaly Detection in Surveillance Videos Based on H265 and Deep Learning.* International Journal of Advanced Technology and Engineering Exploration, Vol 9(92), 2022. DOI: 10.19101/IJATEE.2021.875907

3. Nawaratne, R., De Silva, D., Alahakoon, D., & Yu, X. *Spatiotemporal Anomaly Detection Using Deep Learning for Real-Time Video Surveillance.* IEEE, Federation University Australia.

4. Wenkel, S., Alhazmi, K., Liiv, T., Alrshoud, S., & Simon, M. *Confidence Score: The Forgotten Dimension of Object Detection Performance Evaluation.* Marduk Technologies / KACST.

5. Vasudevan, V. T., Sethy, A., & Ghias, A. R. *Towards Better Confidence Estimation for Neural Models.* University of California San Diego / Alexa AI, Amazon.

6. Lu, C., Shi, J., & Jia, J. *Abnormal Event Detection at 150 FPS in MATLAB.* The Chinese University of Hong Kong. ICCV 2013.

7. Liu, W., Luo, W., Lian, D., & Gao, S. *Future Frame Prediction for Anomaly Detection — A New Baseline.* CVPR 2018.
=======
1. Choudhry et al. — *A Comprehensive Survey of Machine Learning Methods for Surveillance Videos Anomaly Detection*, Deakin University
2. Abbas & Al-Ani — *Anomaly Detection in Surveillance Videos Based on H265 and Deep Learning*, International Journal of Advanced Technology and Engineering Exploration, 2022
3. Nawaratne et al. — *Spatiotemporal Anomaly Detection Using Deep Learning for Real-Time Video Surveillance*, IEEE
4. Wenkel et al. — *Confidence Score: The Forgotten Dimension of Object Detection Performance Evaluation*
5. Vasudevan et al. — *Towards Better Confidence Estimation for Neural Models*, UC San Diego / Amazon Alexa AI
6. Lu et al. — *Abnormal Event Detection at 150 FPS in MATLAB*, The Chinese University of Hong Kong
7. Liu et al. — *Future Frame Prediction for Anomaly Detection — A New Baseline*
