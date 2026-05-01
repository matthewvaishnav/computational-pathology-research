"""
Custom exception classes for HistoCore

Domain-specific exceptions for better error handling and debugging.
"""


# Base exceptions
class HistoCoreError(Exception):
    """Base exception for all HistoCore errors"""
    pass


class ConfigurationError(HistoCoreError):
    """Configuration-related errors"""
    pass


class ValidationError(HistoCoreError):
    """Data validation errors"""
    pass


# Data/IO exceptions
class DataLoadError(HistoCoreError):
    """Failed to load data from disk/network"""
    pass


class DataSaveError(HistoCoreError):
    """Failed to save data to disk/network"""
    pass


class SlideNotFoundError(DataLoadError):
    """WSI slide file not found"""
    pass


class CorruptedDataError(DataLoadError):
    """Data file corrupted or invalid format"""
    pass


# Model exceptions
class ModelError(HistoCoreError):
    """Base exception for model-related errors"""
    pass


class ModelLoadError(ModelError):
    """Failed to load model checkpoint"""
    pass


class ModelInferenceError(ModelError):
    """Error during model inference"""
    pass


class ModelNotFoundError(ModelError):
    """Model checkpoint not found"""
    pass


class UnsupportedModelError(ModelError):
    """Model architecture not supported"""
    pass


# Cache exceptions
class CacheError(HistoCoreError):
    """Base exception for cache operations"""
    pass


class CacheConnectionError(CacheError):
    """Failed to connect to cache backend"""
    pass


class CacheSerializationError(CacheError):
    """Failed to serialize/deserialize cached data"""
    pass


# Database exceptions
class DatabaseError(HistoCoreError):
    """Base exception for database operations"""
    pass


class DatabaseConnectionError(DatabaseError):
    """Failed to connect to database"""
    pass


class DatabaseTransactionError(DatabaseError):
    """Database transaction failed"""
    pass


# Annotation exceptions
class AnnotationError(HistoCoreError):
    """Base exception for annotation operations"""
    pass


class InvalidAnnotationError(AnnotationError):
    """Annotation data invalid or malformed"""
    pass


class AnnotationNotFoundError(AnnotationError):
    """Annotation not found in database"""
    pass


# Active learning exceptions
class ActiveLearningError(HistoCoreError):
    """Base exception for active learning operations"""
    pass


class InsufficientAnnotationsError(ActiveLearningError):
    """Not enough annotations for retraining"""
    pass


class RetrainingError(ActiveLearningError):
    """Model retraining failed"""
    pass


# Federated learning exceptions
class FederatedLearningError(HistoCoreError):
    """Base exception for federated learning"""
    pass


class ClientConnectionError(FederatedLearningError):
    """Failed to connect to federated client"""
    pass


class AggregationError(FederatedLearningError):
    """Model aggregation failed"""
    pass


class PrivacyBudgetExceededError(FederatedLearningError):
    """Differential privacy budget exceeded"""
    pass


# PACS exceptions
class PACSError(HistoCoreError):
    """Base exception for PACS operations"""
    pass


class PACSConnectionError(PACSError):
    """Failed to connect to PACS server"""
    pass


class DICOMError(PACSError):
    """DICOM operation failed"""
    pass


class StudyNotFoundError(PACSError):
    """DICOM study not found"""
    pass


# Streaming exceptions
class StreamingError(HistoCoreError):
    """Base exception for WSI streaming"""
    pass


class TileExtractionError(StreamingError):
    """Failed to extract tile from WSI"""
    pass


class StreamingConnectionError(StreamingError):
    """Streaming connection lost"""
    pass


# Security exceptions
class SecurityError(HistoCoreError):
    """Base exception for security violations"""
    pass


class AuthenticationError(SecurityError):
    """Authentication failed"""
    pass


class AuthorizationError(SecurityError):
    """User not authorized for operation"""
    pass


class IntegrityError(SecurityError):
    """Data integrity check failed"""
    pass


class EncryptionError(SecurityError):
    """Encryption/decryption failed"""
    pass


# Resource exceptions
class ResourceError(HistoCoreError):
    """Base exception for resource issues"""
    pass


class OutOfMemoryError(ResourceError):
    """Insufficient memory for operation"""
    pass


class DiskSpaceError(ResourceError):
    """Insufficient disk space"""
    pass


class GPUError(ResourceError):
    """GPU operation failed"""
    pass


class TimeoutError(ResourceError):
    """Operation timed out"""
    pass


# Threading exceptions
class ThreadingError(HistoCoreError):
    """Base exception for threading issues"""
    pass


class DeadlockError(ThreadingError):
    """Deadlock detected"""
    pass


class ThreadPoolExhaustedError(ThreadingError):
    """Thread pool has no available threads"""
    pass
