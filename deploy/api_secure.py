"""
Secure FastAPI deployment with authentication, rate limiting, and monitoring.

Features:
- JWT-based authentication
- API key authentication
- Rate limiting
- Request validation
- Audit logging
- Metrics export
- CORS configuration
- Security headers
"""

from fastapi import FastAPI, HTTPException, Depends, Security, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict
import torch
import numpy as np
import sys
import os
import time
import hashlib
import secrets
from functools import wraps
import logging
import json

# JWT imports
try:
    from jose import JWTError, jwt
    from passlib.context import CryptContext
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    print("Warning: python-jose not installed. JWT authentication disabled.")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import MultimodalFusionModel, ClassificationHead

# ============================================================================
# Configuration
# ============================================================================

# Security configuration
SECRET_KEY = os.getenv("API_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# API Keys (in production, store in database or secrets manager)
VALID_API_KEYS = {
    os.getenv("API_KEY_1", "demo-key-1"): {"name": "demo-user", "tier": "standard"},
    os.getenv("API_KEY_2", "demo-key-2"): {"name": "premium-user", "tier": "premium"},
}

# Rate limiting configuration
RATE_LIMITS = {
    "standard": {"requests": 100, "window": 3600},  # 100 requests per hour
    "premium": {"requests": 1000, "window": 3600},  # 1000 requests per hour
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Security Utilities
# ============================================================================

if JWT_AVAILABLE:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    if not JWT_AVAILABLE:
        return False
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash password."""
    if not JWT_AVAILABLE:
        raise RuntimeError("JWT not available")
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    if not JWT_AVAILABLE:
        raise RuntimeError("JWT not available")
    
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Dict[str, Any]:
    """Verify JWT token."""
    if not JWT_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="JWT authentication not available"
        )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ============================================================================
# Rate Limiting
# ============================================================================

class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests = defaultdict(list)
    
    def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """Check if request is allowed under rate limit."""
        now = time.time()
        
        # Clean old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if now - req_time < window
        ]
        
        # Check limit
        if len(self.requests[key]) >= limit:
            return False
        
        # Add current request
        self.requests[key].append(now)
        return True
    
    def get_remaining(self, key: str, limit: int, window: int) -> int:
        """Get remaining requests in current window."""
        now = time.time()
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if now - req_time < window
        ]
        return max(0, limit - len(self.requests[key]))

rate_limiter = RateLimiter()

# ============================================================================
# Authentication
# ============================================================================

security = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)) -> Dict[str, Any]:
    """Verify API key."""
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    if api_key not in VALID_API_KEYS:
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return VALID_API_KEYS[api_key]

async def verify_jwt_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict[str, Any]:
    """Verify JWT token."""
    token = credentials.credentials
    return verify_token(token)

async def get_current_user(
    api_key: Optional[str] = Security(api_key_header),
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Dict[str, Any]:
    """Get current user from API key or JWT token."""
    # Try API key first
    if api_key and api_key in VALID_API_KEYS:
        return VALID_API_KEYS[api_key]
    
    # Try JWT token
    if credentials and JWT_AVAILABLE:
        try:
            return verify_token(credentials.credentials)
        except HTTPException:
            pass
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials"
    )

def check_rate_limit(user: Dict[str, Any]):
    """Check rate limit for user."""
    tier = user.get("tier", "standard")
    limits = RATE_LIMITS.get(tier, RATE_LIMITS["standard"])
    
    user_key = user.get("name", "unknown")
    
    if not rate_limiter.is_allowed(user_key, limits["requests"], limits["window"]):
        remaining = rate_limiter.get_remaining(user_key, limits["requests"], limits["window"])
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again later.",
            headers={
                "X-RateLimit-Limit": str(limits["requests"]),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(int(time.time() + limits["window"]))
            }
        )

# ============================================================================
# Request/Response Models
# ============================================================================

class PredictionRequest(BaseModel):
    """Request schema for prediction."""
    wsi_features: Optional[List[List[float]]] = Field(
        None,
        description="WSI patch features [num_patches, 1024]",
        min_items=1,
        max_items=1000
    )
    genomic: Optional[List[float]] = Field(
        None,
        description="Genomic features [2000]",
        min_items=2000,
        max_items=2000
    )
    clinical_text: Optional[List[int]] = Field(
        None,
        description="Tokenized clinical text [seq_len]",
        min_items=1,
        max_items=512
    )
    
    @validator('wsi_features')
    def validate_wsi_features(cls, v):
        if v is not None:
            if not all(len(patch) == 1024 for patch in v):
                raise ValueError("Each WSI patch must have 1024 features")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "wsi_features": [[0.1] * 1024] * 50,
                "genomic": [0.1] * 2000,
                "clinical_text": [100, 200, 300, 400, 500]
            }
        }

class PredictionResponse(BaseModel):
    """Response schema for prediction."""
    predicted_class: int
    confidence: float
    probabilities: List[float]
    available_modalities: List[str]
    request_id: str
    timestamp: str

class TokenRequest(BaseModel):
    """Request schema for token generation."""
    username: str
    password: str

class TokenResponse(BaseModel):
    """Response schema for token."""
    access_token: str
    token_type: str
    expires_in: int

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Secure Computational Pathology API",
    description="Production-ready REST API with authentication and rate limiting",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(",")
)

# Global model storage
MODEL = None
CLASSIFIER = None
DEVICE = None

# Request tracking
request_count = 0
request_times = []

# ============================================================================
# Middleware
# ============================================================================

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    global request_count, request_times
    
    start_time = time.time()
    request_count += 1
    
    # Log request
    logger.info(f"Request {request_count}: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    # Log response time
    process_time = time.time() - start_time
    request_times.append(process_time)
    
    # Keep only last 1000 request times
    if len(request_times) > 1000:
        request_times = request_times[-1000:]
    
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Response {request_count}: {response.status_code} ({process_time:.3f}s)")
    
    return response

# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global MODEL, CLASSIFIER, DEVICE
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Loading model on {DEVICE}...")
    
    # Initialize models
    MODEL = MultimodalFusionModel(embed_dim=256).to(DEVICE)
    CLASSIFIER = ClassificationHead(input_dim=256, num_classes=4).to(DEVICE)
    
    # Load weights if available
    model_path = os.getenv("MODEL_PATH", "models/best_model.pth")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        CLASSIFIER.load_state_dict(checkpoint['task_head_state_dict'])
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.warning("No trained model found, using random weights")
    
    MODEL.eval()
    CLASSIFIER.eval()
    logger.info("Model loaded successfully!")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Shutting down API...")

# ============================================================================
# Public Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Secure Computational Pathology API",
        "version": "2.0.0",
        "status": "running",
        "authentication": "API Key or JWT",
        "endpoints": {
            "/token": "POST - Get JWT token",
            "/predict": "POST - Make predictions (authenticated)",
            "/health": "GET - Health check",
            "/metrics": "GET - API metrics (authenticated)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.post("/token", response_model=TokenResponse)
async def login(request: TokenRequest):
    """
    Get JWT access token.
    
    For demo purposes, accepts any username/password.
    In production, verify against database.
    """
    if not JWT_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="JWT authentication not available. Install python-jose and passlib."
        )
    
    # In production, verify credentials against database
    # For demo, accept any credentials
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": request.username, "tier": "standard"},
        expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

# ============================================================================
# Protected Endpoints
# ============================================================================

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Make prediction from multimodal data (authenticated).
    
    Requires API key or JWT token.
    Subject to rate limiting based on user tier.
    """
    # Check rate limit
    check_rate_limit(user)
    
    if MODEL is None or CLASSIFIER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check at least one modality is provided
    if not any([request.wsi_features, request.genomic, request.clinical_text]):
        raise HTTPException(
            status_code=400,
            detail="At least one modality must be provided"
        )
    
    try:
        # Generate request ID
        request_id = hashlib.sha256(
            f"{user.get('name')}{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Prepare batch
        batch = {}
        available_modalities = []
        
        # WSI features
        if request.wsi_features:
            wsi_tensor = torch.tensor(request.wsi_features, dtype=torch.float32).unsqueeze(0)
            batch['wsi_features'] = wsi_tensor.to(DEVICE)
            available_modalities.append("wsi")
        else:
            batch['wsi_features'] = None
        
        # Genomic features
        if request.genomic:
            genomic_tensor = torch.tensor(request.genomic, dtype=torch.float32).unsqueeze(0)
            batch['genomic'] = genomic_tensor.to(DEVICE)
            available_modalities.append("genomic")
        else:
            batch['genomic'] = None
        
        # Clinical text
        if request.clinical_text:
            clinical_tensor = torch.tensor(request.clinical_text, dtype=torch.long).unsqueeze(0)
            batch['clinical_text'] = clinical_tensor.to(DEVICE)
            available_modalities.append("clinical_text")
        else:
            batch['clinical_text'] = None
        
        # Inference
        with torch.no_grad():
            embeddings = MODEL(batch)
            logits = CLASSIFIER(embeddings)
            probabilities = torch.softmax(logits, dim=-1)
            
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()
            probs_list = probabilities[0].cpu().tolist()
        
        # Log prediction
        logger.info(
            f"Prediction {request_id} by {user.get('name')}: "
            f"class={predicted_class}, confidence={confidence:.3f}"
        )
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probs_list,
            available_modalities=available_modalities,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/metrics")
async def get_metrics(user: Dict[str, Any] = Depends(get_current_user)):
    """Get API metrics (authenticated)."""
    check_rate_limit(user)
    
    avg_response_time = np.mean(request_times) if request_times else 0
    p95_response_time = np.percentile(request_times, 95) if request_times else 0
    
    return {
        "total_requests": request_count,
        "avg_response_time_ms": avg_response_time * 1000,
        "p95_response_time_ms": p95_response_time * 1000,
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/model-info")
async def model_info(user: Dict[str, Any] = Depends(get_current_user)):
    """Get model information (authenticated)."""
    check_rate_limit(user)
    
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    total_params = sum(p.numel() for p in MODEL.parameters()) + \
                   sum(p.numel() for p in CLASSIFIER.parameters())
    
    return {
        "architecture": "MultimodalFusionModel",
        "embed_dim": 256,
        "num_classes": 4,
        "total_parameters": total_params,
        "device": str(DEVICE),
        "supported_modalities": ["wsi", "genomic", "clinical_text"]
    }

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))
    
    logger.info(f"Starting API on {host}:{port} with {workers} workers")
    
    uvicorn.run(
        "api_secure:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True
    )
