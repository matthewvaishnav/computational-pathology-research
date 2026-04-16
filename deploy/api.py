"""
FastAPI deployment example for multimodal pathology model.
Provides REST API endpoints for inference.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import MultimodalFusionModel, ClassificationHead

app = FastAPI(
    title="Computational Pathology API",
    description="REST API for multimodal pathology inference",
    version="1.0.0",
)

# Global model storage
MODEL = None
CLASSIFIER = None
DEVICE = None


class PredictionRequest(BaseModel):
    """Request schema for prediction."""

    wsi_features: Optional[List[List[float]]] = Field(
        None, description="WSI patch features [num_patches, 1024]"
    )
    genomic: Optional[List[float]] = Field(None, description="Genomic features [2000]")
    clinical_text: Optional[List[int]] = Field(
        None, description="Tokenized clinical text [seq_len]"
    )

    class Config:
        schema_extra = {
            "example": {
                "wsi_features": [[0.1] * 1024] * 50,  # 50 patches
                "genomic": [0.1] * 2000,
                "clinical_text": [100, 200, 300, 400, 500],
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction."""

    predicted_class: int
    confidence: float
    probabilities: List[float]
    available_modalities: List[str]


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global MODEL, CLASSIFIER, DEVICE

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {DEVICE}...")

    # Initialize models
    MODEL = MultimodalFusionModel(embed_dim=256).to(DEVICE)
    CLASSIFIER = ClassificationHead(input_dim=256, num_classes=4).to(DEVICE)

    # Load weights if available
    model_path = "models/best_model.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        MODEL.load_state_dict(checkpoint["model_state_dict"])
        CLASSIFIER.load_state_dict(checkpoint["classifier_state_dict"])
        print(f"Loaded model from {model_path}")
    else:
        print("Warning: No trained model found, using random weights")

    MODEL.eval()
    CLASSIFIER.eval()
    print("Model loaded successfully!")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Computational Pathology API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/predict": "POST - Make predictions",
            "/health": "GET - Health check",
            "/model-info": "GET - Model information",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": MODEL is not None, "device": str(DEVICE)}


@app.get("/model-info")
async def model_info():
    """Get model information."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    total_params = sum(p.numel() for p in MODEL.parameters()) + sum(
        p.numel() for p in CLASSIFIER.parameters()
    )

    return {
        "architecture": "MultimodalFusionModel",
        "embed_dim": 256,
        "num_classes": 4,
        "total_parameters": total_params,
        "device": str(DEVICE),
        "supported_modalities": ["wsi", "genomic", "clinical_text"],
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make prediction from multimodal data.

    Accepts WSI features, genomic data, and clinical text.
    At least one modality must be provided.
    """
    if MODEL is None or CLASSIFIER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Check at least one modality is provided
    if not any([request.wsi_features, request.genomic, request.clinical_text]):
        raise HTTPException(
            status_code=400,
            detail="At least one modality (wsi_features, genomic, or clinical_text) must be provided",
        )

    try:
        # Prepare batch
        batch = {}
        available_modalities = []

        # WSI features
        if request.wsi_features:
            wsi_tensor = torch.tensor(request.wsi_features, dtype=torch.float32).unsqueeze(
                0
            )  # [1, num_patches, 1024]
            batch["wsi_features"] = wsi_tensor.to(DEVICE)
            batch["wsi_mask"] = torch.ones(1, wsi_tensor.shape[1], dtype=torch.bool).to(DEVICE)
            available_modalities.append("wsi")
        else:
            batch["wsi_features"] = None
            batch["wsi_mask"] = None

        # Genomic features
        if request.genomic:
            if len(request.genomic) != 2000:
                raise HTTPException(
                    status_code=400,
                    detail=f"Genomic features must have length 2000, got {len(request.genomic)}",
                )
            genomic_tensor = torch.tensor(request.genomic, dtype=torch.float32).unsqueeze(
                0
            )  # [1, 2000]
            batch["genomic"] = genomic_tensor.to(DEVICE)
            available_modalities.append("genomic")
        else:
            batch["genomic"] = None

        # Clinical text
        if request.clinical_text:
            clinical_tensor = torch.tensor(request.clinical_text, dtype=torch.long).unsqueeze(
                0
            )  # [1, seq_len]
            batch["clinical_text"] = clinical_tensor.to(DEVICE)
            batch["clinical_mask"] = torch.ones(1, clinical_tensor.shape[1], dtype=torch.bool).to(
                DEVICE
            )
            available_modalities.append("clinical_text")
        else:
            batch["clinical_text"] = None
            batch["clinical_mask"] = None

        # Inference
        with torch.no_grad():
            embeddings = MODEL(batch)
            logits = CLASSIFIER(embeddings)
            probabilities = torch.softmax(logits, dim=-1)

            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()
            probs_list = probabilities[0].cpu().tolist()

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probs_list,
            available_modalities=available_modalities,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(requests: List[PredictionRequest]):
    """
    Make predictions for multiple samples.

    More efficient than calling /predict multiple times.
    """
    if MODEL is None or CLASSIFIER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(requests) > 32:
        raise HTTPException(status_code=400, detail="Batch size limited to 32 samples")

    results = []
    for req in requests:
        try:
            result = await predict(req)
            results.append(result.dict())
        except HTTPException as e:
            results.append({"error": e.detail})

    return {"predictions": results, "count": len(results)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
