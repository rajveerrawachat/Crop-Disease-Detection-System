"""
Plant Disease Detection Model Module
Uses pretrained EfficientNet-B4 trained on the PlantVillage dataset (38 classes).
Checkpoint: Khawajaa/plant-disease-detector (best_model.pth)
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from huggingface_hub import hf_hub_download

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "best_model.pth"
HF_REPO_ID = "Khawajaa/plant-disease-detector"
HF_FILENAME = "best_model.pth"

# 38 PlantVillage Classes (Preserving exact notebook ordering)
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Standard PlantVillage ImageNet Preprocessing Pipeline
PREPROCESSING_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class EfficientNetB4Classifier(nn.Module):
    """
    EfficientNet-B4 with custom two-layer classification head.
    Matches the architecture in khawaja1447/plant-disease-detector.
    """

    def __init__(self, num_classes: int = 38, dropout: float = 0.4):
        super().__init__()
        backbone = models.efficientnet_b4(weights=None)
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1792, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def ensure_model_file() -> Path:
    """
    Verifies that the model checkpoint exists locally and is non-empty.
    If missing or 0 bytes, downloads it from Hugging Face Hub.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size == 0:
        MODEL_PATH.unlink()

    if not MODEL_PATH.exists():
        print(f"Downloading {HF_FILENAME} from Hugging Face ({HF_REPO_ID})...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            local_dir=str(MODEL_DIR)
        )

    return MODEL_PATH


# Global model cache to avoid re-loading on each inference call
_MODEL_CACHE = None
_DEVICE = None


def get_model(device: torch.device = None) -> Tuple[nn.Module, torch.device]:
    """
    Loads and caches the EfficientNet-B4 model in evaluation mode.
    """
    global _MODEL_CACHE, _DEVICE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE, _DEVICE

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _DEVICE = device

    checkpoint_path = ensure_model_file()
    model = EfficientNetB4Classifier(num_classes=len(CLASS_NAMES), dropout=0.4)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Invalid checkpoint structure in best_model.pth")

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    _MODEL_CACHE = model
    return _MODEL_CACHE, _DEVICE


def format_class_name(class_name: str) -> Tuple[str, str]:
    """
    Splits 'Plant___Condition' into human-readable ('Plant', 'Condition').
    """
    if "___" in class_name:
        plant, condition = class_name.split("___", 1)
    else:
        plant, condition = "Unknown Plant", class_name

    plant = plant.replace("_", " ").strip()
    condition = condition.replace("_", " ").strip()
    return plant, condition


def predict_leaf(image: Image.Image, top_k: int = 5) -> Dict[str, Any]:
    """
    Takes a PIL leaf image and produces structured classification results.

    Returns:
        {
            "predicted_class": str,
            "plant": str,
            "condition": str,
            "confidence": float,
            "image_status": str,
            "top_predictions": [
                {
                    "class": str,
                    "plant": str,
                    "condition": str,
                    "confidence": float,
                    "readable": str
                }, ...
            ]
        }
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a valid PIL.Image instance.")

    model, device = get_model()

    # Preprocess image
    rgb_image = image.convert("RGB")
    input_tensor = PREPROCESSING_TRANSFORM(rgb_image).unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(CLASS_NAMES)), dim=1)

    top_predictions: List[Dict[str, Any]] = []
    for i in range(top_probs.shape[1]):
        idx = top_indices[0][i].item()
        prob = float(top_probs[0][i].item())
        raw_class = CLASS_NAMES[idx]
        plant_name, condition_name = format_class_name(raw_class)
        readable_label = f"{plant_name} — {condition_name}"

        top_predictions.append({
            "class": raw_class,
            "plant": plant_name,
            "condition": condition_name,
            "confidence": prob,
            "readable": readable_label
        })

    top_result = top_predictions[0]
    predicted_class = top_result["class"]
    plant = top_result["plant"]
    condition = top_result["condition"]
    confidence = top_result["confidence"]

    # Qualitative image status
    if "healthy" in condition.lower():
        image_status = "Healthy class predicted"
    else:
        image_status = "Potential disease detected"

    return {
        "predicted_class": predicted_class,
        "plant": plant,
        "condition": condition,
        "confidence": confidence,
        "image_status": image_status,
        "top_predictions": top_predictions
    }