import streamlit as st
st.set_page_config(layout="wide")
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import tempfile
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import warnings

# Suppress OpenBLAS and threading warnings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

# ----------------- CSRNet Model Definition -----------------
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend = self.make_layers([64, 64, 'M', 128, 128, 'M',
                                          256, 256, 256, 'M', 512, 512, 512])
        self.backend = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

    def make_layers(self, cfg, in_channels=3, dilation=False):
        layers = []
        d_rate = 2 if dilation else 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                                   padding=d_rate, dilation=d_rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

# ----------------- Anomaly Classifier for Avenue -----------------
class AnomalyClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.base = models.resnet34(weights="IMAGENET1K_V1")
        self.base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.base.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.base(x)

# ----------------- Load Models -----------------
@st.cache_resource
def load_models():
    csrnet_model = CSRNet()
    state_dict = torch.load("notebooks/csrnet_shanghai.pt", map_location="cpu")
    csrnet_model.load_state_dict(state_dict)
    csrnet_model.eval()

    anomaly_model = AnomalyClassifier(num_classes=3)
    anomaly_model.load_state_dict(torch.load("notebooks/final_op_avenue_model.pt", map_location="cpu"))
    anomaly_model.eval()

    yolo_model = YOLO("notebooks/yolov8m.pt")
    return csrnet_model, anomaly_model, yolo_model

csrnet_model, anomaly_model, yolo_model = load_models()

csr_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

anomaly_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def get_alert_level(count):
    if count <= 5:
        return "Normal", "green"
    elif count <= 10:
        return "Can lead to overcrowding", "gold"
    elif count <= 20:
        return "Possible overcrowding", "orange"
    else:
        return "Overcrowding", "red"

def infer_image(img):
    input_img = csr_transform(img).unsqueeze(0)
    with torch.no_grad():
        output = csrnet_model(input_img)
    density_map = output.squeeze(0).squeeze(0).cpu().numpy()
    count = max(0, density_map.sum())
    return density_map, count

def infer_yolo_fallback(img):
    results = yolo_model(img)
    detections = results[0].boxes.cls.tolist()
    count = detections.count(0)
    return count

def annotate_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(tempfile.gettempdir(), "annotated_video.mp4")
    out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    LABEL2IDX = {"normal": 0, "unusual action": 1, "abnormal object": 2}
    IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}
    COLORS = {'normal': (0, 200, 0), 'unusual action': (255, 140, 0), 'abnormal object': (255, 0, 0)}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
        input_tensor = anomaly_transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            logits = anomaly_model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
            pred_idx = probs.argmax()
            confidence = probs[pred_idx]
            label = IDX2LABEL[pred_idx]
            text = f"{label} ({confidence:.2f})"

        color = COLORS[label]
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        out_writer.write(frame)

    cap.release()
    out_writer.release()
    return out_path

# ----------------- Streamlit UI -----------------
st.title("Surveillance Analysis Dashboard")

tabs = st.tabs(["ShanghaiTech", "Avenue"])

# ShanghaiTech Tab
with tabs[0]:
    st.subheader("Crowd Monitoring")
    mode = st.radio("Select Input Type", ["Image", "Video"], key="shang_mode")

    if mode == "Image":
        image_file = st.file_uploader("Upload a crowd image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="shang_image")

        if image_file is not None:
            img = Image.open(image_file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)

            density_map, pred_count = infer_image(img)
            if pred_count < 1:
                pred_count = infer_yolo_fallback(np.array(img))

            alert_text, alert_color = get_alert_level(pred_count)

            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].imshow(img)
            axs[0].set_title(f"Alert: {alert_text}", color=alert_color)
            axs[0].axis("off")

            axs[1].imshow(density_map, cmap="jet")
            axs[1].set_title(f"Predicted Density\nCount: {int(pred_count)}")
            axs[1].axis("off")

            st.pyplot(fig)

    elif mode == "Video":
        video_file = st.file_uploader("Upload a video (MP4/AVI)", type=["mp4", "avi"], key="shang_video")
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())

            cap = cv2.VideoCapture(tfile.name)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            stframe = st.empty()

            for _ in range(min(frame_count, 100)):
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)

                density_map, pred_count = infer_image(pil_img)
                if pred_count < 1:
                    pred_count = infer_yolo_fallback(rgb_img)

                alert_text, alert_color = get_alert_level(pred_count)

                fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                axs[0].imshow(rgb_img)
                axs[0].set_title(f"Alert: {alert_text}", color=alert_color)
                axs[0].axis("off")

                axs[1].imshow(density_map, cmap="jet")
                axs[1].set_title(f"Predicted Density\nCount: {int(pred_count)}")
                axs[1].axis("off")

                stframe.pyplot(fig)
            cap.release()

# Avenue Tab
with tabs[1]:
    st.subheader("Anomaly Detection")

    video_file = st.file_uploader("Upload a surveillance video (MP4/AVI)", type=["mp4", "avi"], key="avenue_video")

    if video_file is not None:
        st.video(video_file)
        st.write("Processing video and generating results...")

        annotated_video_path = annotate_video(video_file)

        st.success("Annotated video ready:")
        with open(annotated_video_path, "rb") as f:
            st.download_button("Download Annotated Video", f, file_name="anomaly_labeled_avenue.mp4")
        st.video(annotated_video_path)

