"""Neural network model definitions."""

from .stain_normalization import (
    StainNormalizationTransformer,
    PatchEmbedding,
    ColorFeatureEncoder,
    StyleConditioner,
    StyleTransferDecoder
)

from .encoders import (
    WSIEncoder,
    GenomicEncoder,
    ClinicalTextEncoder
)

from .fusion import (
    CrossModalAttention,
    MultiModalFusionLayer
)

from .multimodal import (
    MultimodalFusionModel
)

from .temporal import (
    TemporalAttention,
    CrossSlideTemporalReasoner
)

from .heads import (
    ClassificationHead,
    SurvivalPredictionHead,
    MultiTaskHead
)

from .baselines import (
    SingleModalityModel,
    LateFusionModel,
    AttentionBaseline,
    get_baseline_model
)

__all__ = [
    'StainNormalizationTransformer',
    'PatchEmbedding',
    'ColorFeatureEncoder',
    'StyleConditioner',
    'StyleTransferDecoder',
    'WSIEncoder',
    'GenomicEncoder',
    'ClinicalTextEncoder',
    'CrossModalAttention',
    'MultiModalFusionLayer',
    'MultimodalFusionModel',
    'TemporalAttention',
    'CrossSlideTemporalReasoner',
    'ClassificationHead',
    'SurvivalPredictionHead',
    'MultiTaskHead',
    'SingleModalityModel',
    'LateFusionModel',
    'AttentionBaseline',
    'get_baseline_model'
]
