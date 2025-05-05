# Real-Time Anomaly Detection and Crowd Monitoring

This repository contains an interactive Streamlit app for:

1. **Avenue Dataset** – Person-centric anomaly classification using ResNet34 and YOLOv8.
2. **ShanghaiTech Dataset** – Crowd density estimation using CSRNet with YOLOv8 fallback logic for robust detection.

---

## Project Structure

```
root/
│
├── app.py
├── notebooks/
│   ├── Avenue_Anomaly_Detection.ipynb
│   ├── ShanghaiTech_Overcrowding.ipynb
│   └── Combined_Restructured.ipynb
├── assets/
│   ├── avenue_pipeline.png
│   └── csrnet_pipeline.png
├── README.md
├── requirements.txt
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

   * `yolov8m.pt` → from [Ultralytics](https://github.com/ultralytics/ultralytics#models)
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

## Model Highlights

### ShanghaiTech (CSRNet-Based Crowd Counting)

* **Model Used:** CSRNet with VGG16 frontend and dilated convolutional backend
* **Purpose:** Predicts spatial density maps for crowd estimation from static surveillance images
* **Input:** Static images (`.jpg`) from **ShanghaiTech Part A & B**
* **Ground Truth:** Head annotations in `.mat` files converted to density maps using Gaussian smoothing
* **Output:** Predicted density map with total crowd count
* **Overcrowding Alert:** Triggered when count exceeds **20 people**
* **Evaluation Metrics:**

  * **Part A:** MAE: 207.91, RMSE: 317.36
  * **Part B:** MAE: 31.18, RMSE: 66.59

---

### Avenue Dataset (YOLOv8 + ResNet18-Based Anomaly Classification)

* **Models Used:**

  * **YOLOv8** for detecting people and objects in video frames
  * **ResNet18-based classifier** for anomaly type classification (fine-tuned on Avenue dataset)
* **Input:** Extracted video frames (`.avi → .jpg`)
* **Ground Truth:** Frame-level anomaly labels in `.mat` files
* **Output:** Frame-level class labels:

  * **Normal**
  * **Unusual Action**
  * **Abnormal Object**
* **Visualization:** Annotated videos with bounding boxes and color-coded labels
* **Evaluation Metrics:**
* Original Loss: **92.17**, Improved Loss: **68.76**

---

## Architecture Overview

### ShanghaiTech Crowd Monitoring (CSRNet)

![CSRNet Pipeline](assets/diagram-shanghaitech.png)

### Avenue Anomaly Detection (YOLOv8 + ResNet18)

![Avenue Pipeline](assets/diagram-avenue.png)

---

## Requirements

See `requirements.txt` for full list. Major libraries:

* torch
* torchvision
* streamlit
* opencv-python
* matplotlib
* ultralytics
* pillow
* numpy

---

## Developed By

Stuti Upadhyay — UMBC | DATA 606 Capstone Project
\n Lakshmi Tejaswini Chandra Pampana — UMBC | DATA 606 Capstone Project

---

## References

* Liu, W., Luo, W., Lian, D., & Gao, S. (2018). [**Future Frame Prediction for Anomaly Detection – A New Baseline**](https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Future_Frame_Prediction_CVPR_2018_paper.html). *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

* **A Comprehensive Survey of Machine Learning Methods for Surveillance Videos Anomaly Detection**
  *[https://ieeexplore.ieee.org/document/9655672](https://ieeexplore.ieee.org/document/9655672)*

* **Anomaly Detection in Surveillance Videos based on H.265 and Deep Learning**
  *[https://www.sciencedirect.com/science/article/abs/pii/S0893608021001465](https://www.sciencedirect.com/science/article/abs/pii/S0893608021001465)*

* **Spatiotemporal Anomaly Detection Using Deep Learning for Real-Time Video Surveillance**
  *[https://arxiv.org/abs/2201.01899](https://arxiv.org/abs/2201.01899)*

* **Anomaly Detection Based on Latent Feature Training in Surveillance Scenarios**
  *[https://ieeexplore.ieee.org/document/9813961](https://ieeexplore.ieee.org/document/9813961)*

* **Detecting Abnormal Activities with Spatiotemporal Features**
  *[https://openaccess.thecvf.com/content\_iccv\_2013/html/Mehran\_Abnormal\_Crowd\_Behavior\_2013\_ICCV\_paper.html](https://openaccess.thecvf.com/content_iccv_2013/html/Mehran_Abnormal_Crowd_Behavior_2013_ICCV_paper.html)*

---
