"""Multiple Instance Learning (MIL) models."""

from src.models.mil.attention_mil import *
from src.models.mil.clam import *
from src.models.mil.instance_clustering import *
from src.models.mil.mil_base import *
from src.models.mil.nnmil import *
from src.models.mil.transmil import *

__all__ = [
    "attention_mil",
    "clam",
    "instance_clustering",
    "mil_base",
    "nnmil",
    "transmil",
]
