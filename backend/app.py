import os
import json
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# =====================================================
# 1. FastAPI app
# =====================================================
app = FastAPI()

# =====================================================
# 2. Modelo CNN ligero
# =====================================================
class EyeMouthCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # open, closed, half, yawn
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EyeMouthCNN().to(device)
model.eval()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# =====================================================
# 3. MediaPipe Face Mesh (con fallback para Render)
# =====================================================
mp_face = None

try:
    if hasattr(mp, "solutions"):
        mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("MediaPipe FaceMesh inicializado correctamente")
    else:
        print("MediaPipe instalado sin soporte FaceMesh (Render / Py3.13)")
except Exception as e:
    print("Error inicializando MediaPipe:", e)
    mp_face = None


# =====================================================
# 4. Aprendizaje incremental
# =====================================================
train_buffer = deque(maxlen=100)

def fine_tune_model():
    if len(train_buffer) < 10:
        return
    model.train()
    for img, label in train_buffer:
        img_t = transform(img).unsqueeze(0).to(device)
        lbl_t = torch.tensor([label]).to(device)
        optimizer.zero_grad()
        out = model(img_t)
        loss = criterion(out, lbl_t)
        loss.backward()
        optimizer.step()
    model.eval()

# =====================================================
# 5. Utilidades
# =====================================================
def base64_to_image(b64):
    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    return np.array(img)

def extract_roi(image, landmarks, idxs):
    h, w, _ = image.shape
    xs = [int(landmarks[i].x * w) for i in idxs]
    ys = [int(landmarks[i].y * h) for i in idxs]
    return image[min(ys):max(ys), min(xs):max(xs)]

def predict(roi):
    if roi is None or roi.size == 0:
        return None
    img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.argmax(model(img_t), dim=1).item()

# =====================================================
# 6. WebSocket FastAPI
# =====================================================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = json.loads(await ws.receive_text())
            frame = base64_to_image(data["frame"])
            feedback = data.get("feedback")

            results = None
            if mp_face is not None:
                results = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            response = {"left": None, "right": None, "mouth": None}

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                left = extract_roi(frame, lm, [33, 133])
                right = extract_roi(frame, lm, [362, 263])
                mouth = extract_roi(frame, lm, [78, 308])

                response["left"] = predict(left)
                response["right"] = predict(right)
                response["mouth"] = predict(mouth)

                if feedback:
                    if "left" in feedback:
                        train_buffer.append((Image.fromarray(left), feedback["left"]))
                    if "right" in feedback:
                        train_buffer.append((Image.fromarray(right), feedback["right"]))
                    if "mouth" in feedback:
                        train_buffer.append((Image.fromarray(mouth), feedback["mouth"]))
                    fine_tune_model()

            await ws.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print("Cliente desconectado")
