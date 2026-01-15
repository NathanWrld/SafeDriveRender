import asyncio
import websockets
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import mediapipe as mp
from collections import deque

# -----------------------------
# 1. Modelo CNN ligero
# -----------------------------
class EyeMouthCNN(nn.Module):
    def __init__(self):
        super(EyeMouthCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*16*16, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 0=open, 1=closed, 2=half-open, 3=yawn
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EyeMouthCNN().to(device)
model.eval()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

# -----------------------------
# 2. Mediapipe Face Mesh
# -----------------------------
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, 
                                          max_num_faces=1,
                                          refine_landmarks=True,
                                          min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)

# -----------------------------
# 3. Buffer para aprendizaje incremental
# -----------------------------
buffer_size = 100
train_buffer = deque(maxlen=buffer_size)

def fine_tune_model():
    if len(train_buffer) < 10:
        return
    model.train()
    for img, label in train_buffer:
        img_tensor = transform(img).unsqueeze(0).to(device)
        label_tensor = torch.tensor([label]).to(device)
        optimizer.zero_grad()
        output = model(img_tensor)
        loss = criterion(output, label_tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    print("Modelo ajustado con aprendizaje incremental.")

# -----------------------------
# 4. Funciones auxiliares
# -----------------------------
def base64_to_image(b64_string):
    b = base64.b64decode(b64_string)
    img = Image.open(BytesIO(b)).convert('RGB')
    return np.array(img)

def extract_eye_mouth_ROI(image, landmarks):
    h, w, _ = image.shape
    left_eye_idx = [33, 133]
    right_eye_idx = [362, 263]
    mouth_idx = [78, 308]

    def get_box(idx_list):
        xs = [int(landmarks[i].x * w) for i in idx_list]
        ys = [int(landmarks[i].y * h) for i in idx_list]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return image[y_min:y_max, x_min:x_max]

    left_eye = get_box(left_eye_idx)
    right_eye = get_box(right_eye_idx)
    mouth = get_box(mouth_idx)
    return left_eye, right_eye, mouth

def predict_state(roi):
    if roi is None or roi.size == 0:
        return None
    img_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_tensor)
        pred = torch.argmax(out, dim=1).item()
    return pred

# -----------------------------
# 5. WebSocket Server
# -----------------------------
async def handler(websocket, path):
    async for message in websocket:
        try:
            # Se espera mensaje JSON con "frame" (base64) y opcional "feedback"
            import json
            data = json.loads(message)
            frame_b64 = data.get("frame")
            feedback = data.get("feedback")  # opcional: {"left":0,...}

            frame = base64_to_image(frame_b64)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face.process(frame_rgb)

            response = {"left":None, "right":None, "mouth":None}

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                left_eye, right_eye, mouth = extract_eye_mouth_ROI(frame, landmarks)
                left_state = predict_state(left_eye)
                right_state = predict_state(right_eye)
                mouth_state = predict_state(mouth)

                response["left"] = left_state
                response["right"] = right_state
                response["mouth"] = mouth_state

                # Aprendizaje incremental si el usuario valida feedback
                if feedback:
                    if "left" in feedback:
                        train_buffer.append((Image.fromarray(cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB)), feedback["left"]))
                    if "right" in feedback:
                        train_buffer.append((Image.fromarray(cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)), feedback["right"]))
                    if "mouth" in feedback:
                        train_buffer.append((Image.fromarray(cv2.cvtColor(mouth, cv2.COLOR_BGR2RGB)), feedback["mouth"]))
                    fine_tune_model()

            await websocket.send(json.dumps(response))
        except Exception as e:
            print("Error:", e)
            await websocket.send(json.dumps({"error": str(e)}))

# -----------------------------
# 6. Ejecutar servidor
# -----------------------------
start_server = websockets.serve(handler, "0.0.0.0", 8765)
print("Servidor WebSocket iniciado en ws://localhost:8765")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
