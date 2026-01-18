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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware

# =====================================================
# 1. FastAPI app
# =====================================================
app = FastAPI()

# ===================== CORS (OBLIGATORIO) =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # luego puedes limitar
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# 3. MediaPipe (placeholder)
# =====================================================
print("MediaPipe instalado sin soporte FaceMesh (Render / Py3.13)")

# =====================================================
# 4. Aprendizaje incremental
# =====================================================
train_buffer = deque(maxlen=200)

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

def predict(roi):
    if roi is None or roi.size == 0:
        return None
    img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.argmax(model(img_t), dim=1).item()

# =====================================================
# 6. Endpoint HTTP base
# =====================================================
@app.get("/")
async def root():
    return {"status": "Backend activo"}

# =====================================================
# 7. Endpoint que FALTABA (FRONTEND → BACKEND)
# =====================================================
@app.post("/api/detection-event")
async def detection_event(data: dict):
    """
    Aquí llegan los datos desde detection.js
    """
    print("📥 Evento recibido:", data)

    # Punto ideal para:
    # - guardar en BD
    # - ajustar umbrales
    # - activar aprendizaje futuro

    return {"status": "ok"}

# =====================================================
# 8. WebSocket FastAPI (aprendizaje incremental)
# =====================================================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = json.loads(await ws.receive_text())
            frame = base64_to_image(data["frame"])
            feedback = data.get("feedback")

            response = {"left": None, "right": None, "mouth": None}

            if feedback:
                for key in ["left", "right", "mouth"]:
                    if key in feedback:
                        train_buffer.append(
                            (Image.fromarray(frame), feedback[key])
                        )
                fine_tune_model()

            await ws.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print("🔌 Cliente desconectado")
