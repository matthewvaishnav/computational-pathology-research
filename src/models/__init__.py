"""Neural network model definitions."""

from src.models.mil.attention_mil import CLAM, AttentionMIL
from src.models.baselines import (
    AttentionBaseline,
    LateFusionModel,
    SingleModalityModel,
    get_baseline_model,
)
from src.models.components.encoders import ClinicalTextEncoder, GenomicEncoder, WSIEncoder
from src.models.foundation import (
    CONCHEncoder,
    FeatureProjector,
    FoundationModelEncoder,
    PhikonEncoder,
    UNIEncoder,
    load_foundation_model,
)
from src.models.foundation_adapter import FoundationModelAdapter
from src.models.components.fusion import CrossModalAttention, MultiModalFusionLayer
from src.models.components.heads import ClassificationHead, MultiTaskHead, SurvivalPredictionHead
from src.models.mil.instance_clustering import (
    CLAMInstanceBranch,
    InstanceClusteringModule,
    cluster_instances,
)
from src.models.mil.mil_base import MILBase
from src.models.multimodal import MultimodalFusionModel
from src.models.mil.nnmil import nnMIL
from src.models.pretrained import (
    PretrainedFeatureExtractor,
    create_wsi_encoder_with_pretrained,
    get_recommended_model,
    list_pretrained_models,
)
from src.models.stain_normalization import (
    ColorFeatureEncoder,
    PatchEmbedding,
    StainNormalizationTransformer,
    StyleConditioner,
    StyleTransferDecoder,
)
from src.models.temporal import CrossSlideTemporalReasoner, TemporalAttention
from src.models.mil.transmil import TransMIL


def __getattr__(name):
    """Lazy import for ResNetFeatureExtractor to avoid eager torchvision import."""
    if name == "ResNetFeatureExtractor":
        from src.models.components.feature_extractors import ResNetFeatureExtractor

        return ResNetFeatureExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AttentionMIL",
    "CLAM",
    "nnMIL",
    "TransMIL",
    "FoundationModelAdapter",
    "StainNormalizationTransformer",
    "PatchEmbedding",
    "ColorFeatureEncoder",
    "StyleConditioner",
    "StyleTransferDecoder",
    "WSIEncoder",
    "GenomicEncoder",
    "ClinicalTextEncoder",
    "CrossModalAttention",
    "MultiModalFusionLayer",
    "MultimodalFusionModel",
    "TemporalAttention",
    "CrossSlideTemporalReasoner",
    "ClassificationHead",
    "SurvivalPredictionHead",
    "MultiTaskHead",
    "SingleModalityModel",
    "LateFusionModel",
    "AttentionBaseline",
    "get_baseline_model",
    "ResNetFeatureExtractor",
    "PretrainedFeatureExtractor",
    "create_wsi_encoder_with_pretrained",
    "get_recommended_model",
    "list_pretrained_models",
    "FoundationModelEncoder",
    "PhikonEncoder",
    "UNIEncoder",
    "CONCHEncoder",
    "load_foundation_model",
    "FeatureProjector",
    "InstanceClusteringModule",
    "CLAMInstanceBranch",
    "cluster_instances",
    "MILBase",
]
