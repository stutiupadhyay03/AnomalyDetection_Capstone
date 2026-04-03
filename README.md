
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
└── README.md
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/DATA606_Capstone_AnomalyDetection.git
cd DATA606_Capstone_AnomalyDetection
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### 3. Install dependencies

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

```bash
streamlit run app.py
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

---

## Developed By

- **Stuti Upadhyay** — UMBC | DATA 606 Capstone
- **Lakshmi Tejaswini Chandra Pampana** — UMBC | DATA 606 Capstone

---

## References

1. Choudhry, N., Abawajy, J., Huda, S., & Rao, I. *A Comprehensive Survey of Machine Learning Methods for Surveillance Videos Anomaly Detection.* Deakin University.

2. Abbas, Z. K., & Al-Ani, A. A. *Anomaly Detection in Surveillance Videos Based on H265 and Deep Learning.* International Journal of Advanced Technology and Engineering Exploration, Vol 9(92), 2022. DOI: 10.19101/IJATEE.2021.875907

3. Nawaratne, R., De Silva, D., Alahakoon, D., & Yu, X. *Spatiotemporal Anomaly Detection Using Deep Learning for Real-Time Video Surveillance.* IEEE, Federation University Australia.

4. Wenkel, S., Alhazmi, K., Liiv, T., Alrshoud, S., & Simon, M. *Confidence Score: The Forgotten Dimension of Object Detection Performance Evaluation.* Marduk Technologies / KACST.

5. Vasudevan, V. T., Sethy, A., & Ghias, A. R. *Towards Better Confidence Estimation for Neural Models.* University of California San Diego / Alexa AI, Amazon.

6. Lu, C., Shi, J., & Jia, J. *Abnormal Event Detection at 150 FPS in MATLAB.* The Chinese University of Hong Kong. ICCV 2013.

7. Liu, W., Luo, W., Lian, D., & Gao, S. *Future Frame Prediction for Anomaly Detection — A New Baseline.* CVPR 2018.
