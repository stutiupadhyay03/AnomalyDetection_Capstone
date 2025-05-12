
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
├── app.py                               # Streamlit dashboard combining both modules
├── notebooks/                           # Jupyter notebooks for each dataset
│   ├── Avenue_Anomaly_Detection.ipynb   # YOLOv8 + ResNet18 for anomaly detection
│   ├── ShanghaiTech_Overcrowding.ipynb  # CSRNet-based crowd estimation
│
├── assets/                              # Visual diagrams for architecture explanation
│   ├── avenue_pipeline.png              # Anomaly detection pipeline (Avenue)
│   └── csrnet_pipeline.png              # Crowd monitoring pipeline (ShanghaiTech)
│
├── references/                          # Supporting research papers in PDF format
│   ├── *.pdf                            # Cited academic and technical papers
│
├── requirements.txt                     # All Python dependencies for local/app deployment
├── README.md                            # Project overview and instructions

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

## 🔗 Live Demo

Try the real-time surveillance dashboard hosted on Streamlit Cloud:
**[Click here to launch the app](https://anomalydetectioncapstone-aagua27sjxluyvjretnrul.streamlit.app)**

Features:

* Upload crowd images or videos and receive density maps and overcrowding alerts (ShanghaiTech)
* Upload surveillance footage to get anomaly labels with annotated video output (Avenue Dataset)

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

---

## References


1. **A Comprehensive Survey of Machine Learning Methods for Surveillance Videos Anomaly Detection**
   Nomica Choudhry, Jemal Abawajy, Shamsul Huda, and Imran Rao
   Faculty of Science, Engineering and Built Environment, Deakin University, Australia
   Department of Computer Science, NUML, Pakistan
   Blue Brackets Technologies, Islamabad, Pakistan
   *Corresponding author:* [choudhryn@deakin.edu.au](mailto:choudhryn@deakin.edu.au)

2. **Anomaly Detection in Surveillance Videos Based on H265 and Deep Learning**
   Zainab K. Abbas and Ayad A. Al-Ani
   Department of Information and Communication Engineering, Al-Nahrain University, Baghdad, Iraq
   *Published in:* International Journal of Advanced Technology and Engineering Exploration, Vol 9(92), 2022
   *DOI:* 10.19101/IJATEE.2021.875907

3. **Spatiotemporal Anomaly Detection Using Deep Learning for Real-Time Video Surveillance**
   Rashmika Nawaratne, Daswin De Silva, Damminda Alahakoon, Xinghuo Yu
   *Affiliations:* IEEE Members, Federation University Australia

4. **Confidence Score: The Forgotten Dimension of Object Detection Performance Evaluation**
   Simon Wenkel, Khaled Alhazmi, Tanel Liiv, Saud Alrshoud, Martin Simon
   *Corresponding author:* [khazmi@kacst.edu.sa](mailto:khazmi@kacst.edu.sa)
   *Affiliations:* Marduk Technologies, Saudi Arabia, KACST

5. **Towards Better Confidence Estimation for Neural Models**
   Vishal Thanvantri Vasudevan, Abhinav Sethy, Alireza Roshan Ghias
   *Affiliations:* University of California, San Diego and Alexa AI, Amazon

6. **Abnormal Event Detection at 150 FPS in MATLAB**
   Cewu Lu, Jianping Shi, Jiaya Jia
   The Chinese University of Hong Kong
   *Emails:* {cwlu, jpshi, leojia}@cse.cuhk.edu.hk
   
7. **Future Frame Prediction for Anomaly Detection -- A New Baseline**
   W. Liu and W. Luo and D. Lian and S. Gao
   

---
