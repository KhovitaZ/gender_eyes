import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from datetime import datetime
import os

# --- Konfigurasi model ---
class_names = ["Female", "Male"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchvision.models import mobilenet_v3_small

def build_model():
    model = mobilenet_v3_small(pretrained=False)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    return model

model = build_model()
model.load_state_dict(torch.load("gender_eye_model.pth", map_location=device))
model.to(device)
model.eval()

# --- Transformasi gambar ---
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# --- Deteksi mata ---
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# --- Fungsi prediksi ---
def predict_gender(eye_crop):
    eye_pil = Image.fromarray(cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB))
    img_tensor = transform(eye_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
        pred = int(prob > 0.5)
        confidence = prob if pred == 1 else 1 - prob
    return class_names[pred], confidence

# --- Streamlit UI ---
st.set_page_config(page_title="Gender Classification", layout="centered")
st.title("Gender Classification from Eye")

# Tab layout
tab1, tab2, tab3 = st.tabs(["📹 Realtime Webcam", "📸 Kamera", "📁 Upload File"])

# --- Tab 1: Realtime Webcam ---
with tab1:
    run = st.checkbox("Start Realtime Detection")
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Gagal membaca dari kamera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

            for (ex, ey, ew, eh) in eyes[:1]:
                y1 = max(0, ey - int(0.3 * eh) - 15)
                y2 = ey + eh
                x1 = max(0, ex)
                x2 = ex + ew
                eye_crop = frame[y1:y2, x1:x2]

                gender, conf = predict_gender(eye_crop)
                label = f"{gender} ({conf:.2%})"
                color = (0, 255, 0) if gender == "Male" else (255, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

# --- Fungsi bantu untuk tab 2 dan 3 ---
def handle_image_input(image_source):
    st.image(image_source, caption="Input Image", use_column_width=True)

    open_cv_image = np.array(image_source)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)

    if len(eyes) == 0:
        st.error("❌ Tidak terdeteksi mata.")
    else:
        for (ex, ey, ew, eh) in eyes[:1]:
            y1 = max(0, ey - int(0.3 * eh) - 30)
            y2 = ey + eh
            x1 = max(0, ex)
            x2 = ex + ew

            eye_crop = open_cv_image[y1:y2, x1:x2]
            eye_image = Image.fromarray(cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB))
            st.image(eye_image, caption="Cropped Eye + Eyebrow", width=224)

            img_tensor = transform(eye_image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.sigmoid(output).item()
                pred = int(prob > 0.5)
                confidence = prob if pred == 1 else 1 - prob

            st.markdown(f"### 🧠 Prediction: **{class_names[pred]}**")
            st.markdown(f"Confidence: `{confidence:.2%}`")

            if confidence < 0.6:
                st.warning("⚠️ Model kurang yakin terhadap prediksi ini.")

# Fungsi dengan tombol save
def handle_image_input_with_save(image_source):
    handle_image_input(image_source)  # panggil fungsi utama tanpa save dulu
    
    # karena handle_image_input sudah menampilkan gambar dan prediksi,
    # kita hanya perlu menambahkan tombol save di sini:

    # ambil eye_image dan prediksi dari image_source seperti di atas, supaya bisa save
    # agar tidak duplikasi, kita bisa pindahkan kode crop dan prediksi ke fungsi helper
    # atau untuk sederhana, ulangi crop dan prediksi di sini:

    open_cv_image = np.array(image_source)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)

    if len(eyes) > 0:
        ex, ey, ew, eh = eyes[0]
        y1 = max(0, ey - int(0.3 * eh) - 30)
        y2 = ey + eh
        x1 = max(0, ex)
        x2 = ex + ew
        eye_crop = open_cv_image[y1:y2, x1:x2]
        eye_image = Image.fromarray(cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB))

        img_tensor = transform(eye_image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()
            pred = int(prob > 0.5)

        if st.button("💾 Save Eye Image", key=f"save_eye_{datetime.now().strftime('%Y%m%d%H%M%S')}"):
            save_base = "save_gambar"
            class_folder = class_names[pred].lower()
            save_dir = os.path.join(save_base, class_folder)
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{class_folder}_eye_{timestamp}.jpg"
            save_path = os.path.join(save_dir, filename)
            eye_image.save(save_path)
            st.success(f"Gambar disimpan di: `{save_path}`")

# --- Tab 2: Ambil Gambar dari Kamera ---
with tab2:
    st.write("Gunakan tombol di bawah untuk mengaktifkan atau menonaktifkan kamera.")

    if "camera_on" not in st.session_state:
        st.session_state.camera_on = False

    if st.button("📷 Aktifkan / Nonaktifkan Kamera"):
        st.session_state.camera_on = not st.session_state.camera_on

    if st.session_state.camera_on:
        camera_image = st.camera_input("Ambil gambar wajah:")
        if camera_image is not None:
            image_source = Image.open(camera_image).convert("RGB")
            handle_image_input_with_save(image_source)  # Panggil versi dengan tombol save
    else:
        st.info("📌 Kamera tidak aktif. Tekan tombol di atas untuk mengaktifkan.")

# Di Tab 3 (upload file)
with tab3:
    st.write("Unggah gambar wajah untuk klasifikasi.")
    uploaded_file = st.file_uploader("Unggah gambar:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_source = Image.open(uploaded_file).convert("RGB")
        handle_image_input(image_source)
