
# Real-Time Anomaly Detection and Crowd Monitoring

This repository contains an interactive Streamlit app for:

1. **Avenue Dataset** – Person-centric anomaly classification using ResNet18 and YOLOv8.
2. **ShanghaiTech Dataset** – Crowd density estimation using CSRNet with YOLOv8 fallback logic for robust detection.

---

## Motivation and Backgroud

1. Manual Monitoring Limitations
Traditional surveillance systems heavily depend on human operators to continuously monitor camera feeds for suspicious activity. However, this manual approach is prone to human errors such as fatigue, distraction, or delayed response times. These limitations can lead to false alarms, missed anomalies, and overall inefficiency in real-time decision-making, especially during critical events or peak activity hours.
2. Need for Intelligent Surveillance in an Automated World
As we advance toward an era where even household appliances are becoming smart and interconnected, it is imperative that our public safety systems evolve as well. Modern surveillance must keep pace with technological trends, adopting intelligent automation to handle large volumes of video data efficiently, minimize human dependency, and ensure prompt and accurate threat detection.
3. Project Goal: Automated Anomaly Detection in Real-World Surveillance
This project aims to develop an AI-powered system capable of automatically detecting and classifying anomalous events in surveillance footage. These include situations such as overcrowding in public spaces, sudden or unusual movements, and the presence of unexpected or prohibited objects (e.g., bicycles in pedestrian zones, abandoned bags). By integrating deep learning models into a real-time interface, the solution offers scalability, accuracy, and adaptability across diverse surveillance environments.
---

## Project Structure

```
root/
│
├── app.py                              # Streamlit interface
├── notebooks/
│   ├── Avenue_Anomaly_Detection.ipynb  # Anomaly detection and classification
│   ├── ShanghaiTech_Overcrowding.ipynb# Crowd counting and alerts
│   └── Combined_Restructured.ipynb    # Unified notebook
├── assets/
│   ├── avenue_pipeline.png             # Architecture diagram (Avenue)
│   └── csrnet_pipeline.png             # Architecture diagram (ShanghaiTech)
├── README.md                           # Project documentation
├── requirements.txt                    # All required Python libraries
```

---

## How to Run

1. **Clone the repository**

```bash
git clone https://github.com/your-username/DATA606_Capstone_AnomalyDetection.git
cd DATA606_Capstone_AnomalyDetection
```

2. **(Optional) Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install all dependencies**

```bash
pip install -r requirements.txt
```

4. **Download required model files**

* `yolov8m.pt` → from Ultralytics
* `csrnet_shanghai.pt` → pretrained CSRNet model
* `final_op_avenue_model.pt` → trained ResNet18 classifier for Avenue

> Place these files in the **root** directory of the project.

5. **Run the app**

```bash
streamlit run app.py
```

6. **View in browser**

Go to: `http://localhost:8501`

---

## Datasets Used

### 1. ShanghaiTech Dataset

**Purpose**
Originally designed for crowd counting and density estimation. Often adapted for anomaly detection and overcrowding alerts in surveillance.

**Structure**

* Part A – Dense crowds (e.g., Shanghai city streets)

  * Train: 300 annotated images
  * Test: 182 images
* Part B – Sparse crowds (e.g., parks)

  * Train: 400 annotated images
  * Test: 316 images

**Annotations**
Ground truth stored in `.mat` files containing head coordinates, converted to Gaussian density maps.

---

### 2. Avenue Dataset

**Purpose**
Designed for video anomaly detection in surveillance scenarios. Focuses on identifying unusual behaviors like running, throwing, loitering, or carrying strange objects.

**Structure**

* 16 training videos (normal behavior only)
* 21 test videos (contains anomalies)
* Resolution: 640×360 at \~25 FPS
* \~15,328 test frames

**Visual Context**
All scenes recorded from a static camera on a single pedestrian avenue in a controlled indoor/outdoor hybrid setting.

**Anomalies Include**

* People running
* Moving in abnormal directions
* Throwing items
* Carrying abnormal objects (bikes, bags, carts, etc.)

---

## Model Highlights

### ShanghaiTech (CSRNet-Based Crowd Counting)

* **Model:** CSRNet with VGG16 frontend and dilated convolutional backend
* **Task:** Generate density map and predict head count in static images
* **Input:** `.jpg` images
* **Ground Truth:** Annotated `.mat` files with head positions
* **Output:** Density map + total count
* **Overcrowding Alert:** Triggered if count exceeds 20
* **Metrics:**

  * Part A → MAE: 207.91, RMSE: 317.36
  * Part B → MAE: 31.18, RMSE: 66.59

---

### Avenue Dataset (YOLOv8 + ResNet18-Based Anomaly Classification)

* **Object Detection:** YOLOv8 to detect people and objects
* **Classification:** ResNet18 frame classifier fine-tuned to classify anomalies
* **Input:** Video frames extracted from `.avi` files
* **Labels:** Normal / Unusual Action / Abnormal Object
* **Visualization:** Bounding boxes + class label overlay
* **Performance:**

  * Original Loss: 92.17
  * Improved Loss: 68.76

---

## Architecture Overview

### ShanghaiTech Crowd Monitoring (CSRNet)

![CSRNet Pipeline](assets/diagram-shanghaitech.png)

### Avenue Anomaly Detection (YOLOv8 + ResNet18)

![Avenue Pipeline](assets/diagram-avenue.png)

---

## Requirements

* torch
* torchvision
* streamlit
* opencv-python
* matplotlib
* ultralytics
* pillow
* numpy

See `requirements.txt` for full list of packages and versions.

---

## Future Limitations & Directions

### 1. Real-Time Video Stream Integration

* Integrate CCTV or webcam feeds using OpenCV/RTSP for active surveillance.
* Requires reliable edge devices and live networking, currently out of scope.

### 2. Add Temporal Awareness to ResNet

* Future versions could integrate LSTM/GRU/Transformer after ResNet for motion context.
* This adds complexity and requires GPU resources, not suitable for current CPU-only real-time constraints.

### 3. Extend Anomaly Classes

* Expand classification beyond three current labels to include:

  * Fighting/aggression
  * Climbing over barriers
  * Tailgating
  * Bicycles in pedestrian zones
  * Unauthorized group gatherings
* Requires extensive relabeling and retraining with balanced datasets.

### 4. Alert Notification System

* Enable email, SMS, or dashboard-based alerting for anomalies and overcrowding.
* Would require backend infrastructure like Twilio, SendGrid, or cloud APIs.

### 5. Edge and Cloud Deployment

* Use Jetson devices for real-time edge inference.
* Deploy on AWS, Azure, or GCP for scalable monitoring, APIs, and dashboards.

---

## Developed By

* Stuti Upadhyay — UMBC | DATA 606 Capstone Project
* Lakshmi Tejaswini Chandra Pampana — UMBC | DATA 606 Capstone Project

---

## References

* Liu, W., Luo, W., Lian, D., & Gao, S. *Future Frame Prediction for Anomaly Detection – A New Baseline*. IEEE CVPR, 2018.

* *Anomaly Detection Based on Latent Feature Training in Surveillance Scenarios*

* *Spatiotemporal Anomaly Detection Using Deep Learning for Real-Time Video Surveillance*

* *Anomaly Detection in Surveillance Videos Based on H.265 and Deep Learning*

* *A Comprehensive Survey of Machine Learning Methods for Surveillance Videos Anomaly Detection*

* *Abnormal Event Detection in Crowded Scenes Using Sparse Representation*

---
