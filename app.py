# FastAPI application for chest X-ray pneumonia classification

import torch
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from torchvision import transforms

from xray.constant.training_pipeline import (
    DEVICE,
    PREDICTION_LABEL,
)
from xray.ml.model.arch import Net


app = FastAPI(
    title="PneumoVision API",
    description="Chest X-ray Pneumonia Classification API",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# -------------------------------------------------
# Device
# -------------------------------------------------
device = DEVICE


# -------------------------------------------------
# Load trained model
# -------------------------------------------------
model = Net().to(device)

checkpoint = torch.load(
    "lung_disease_cnn_final.pth",
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# -------------------------------------------------
# Deterministic inference preprocessing
# Must match the finalized test/inference transform
# -------------------------------------------------
inference_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)


# -------------------------------------------------
# Health endpoint
# -------------------------------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "active_page": "home"})


@app.get("/predict-ui")
def predict_ui(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request, "active_page": "analyze"})


@app.get("/about")
def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request, "active_page": "about"})


@app.get("/technology")
def technology(request: Request):
    return templates.TemplateResponse("technology.html", {"request": request, "active_page": "technology"})


@app.get("/pneumonia-info")
def pneumonia_info(request: Request):
    return templates.TemplateResponse("info.html", {"request": request, "active_page": "info"})


# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    # Open uploaded image
    image = Image.open(
        file.file
    ).convert("RGB")

    # Apply inference preprocessing
    input_tensor = (
        inference_transform(image)
        .unsqueeze(0)
        .to(device)
    )

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)

        prediction_index = torch.argmax(
            outputs,
            dim=1
        ).item()

    prediction_label = PREDICTION_LABEL[
        prediction_index
    ]

    return {
        "prediction_index": prediction_index,
        "prediction_label": prediction_label,
    }
