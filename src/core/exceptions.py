"""
Custom exception classes for HistoCore

Domain-specific exceptions for better error handling and debugging.
"""


# Base exceptions
class HistoCoreError(Exception):
    """Base exception for all HistoCore errors"""



class ConfigurationError(HistoCoreError):
    """Configuration-related errors"""



class ValidationError(HistoCoreError):
    """Data validation errors"""



# Data/IO exceptions
class DataLoadError(HistoCoreError):
    """Failed to load data from disk/network"""



class DataSaveError(HistoCoreError):
    """Failed to save data to disk/network"""



class SlideNotFoundError(DataLoadError):
    """WSI slide file not found"""



class CorruptedDataError(DataLoadError):
    """Data file corrupted or invalid format"""



# Model exceptions
class ModelError(HistoCoreError):
    """Base exception for model-related errors"""



class ModelLoadError(ModelError):
    """Failed to load model checkpoint"""



class ModelInferenceError(ModelError):
    """Error during model inference"""



class ModelNotFoundError(ModelError):
    """Model checkpoint not found"""



class UnsupportedModelError(ModelError):
    """Model architecture not supported"""



# Cache exceptions
class CacheError(HistoCoreError):
    """Base exception for cache operations"""



class CacheConnectionError(CacheError):
    """Failed to connect to cache backend"""



class CacheSerializationError(CacheError):
    """Failed to serialize/deserialize cached data"""



# Database exceptions
class DatabaseError(HistoCoreError):
    """Base exception for database operations"""



class DatabaseConnectionError(DatabaseError):
    """Failed to connect to database"""



class DatabaseTransactionError(DatabaseError):
    """Database transaction failed"""



# Annotation exceptions
class AnnotationError(HistoCoreError):
    """Base exception for annotation operations"""



class InvalidAnnotationError(AnnotationError):
    """Annotation data invalid or malformed"""



class AnnotationNotFoundError(AnnotationError):
    """Annotation not found in database"""



# Active learning exceptions
class ActiveLearningError(HistoCoreError):
    """Base exception for active learning operations"""



class InsufficientAnnotationsError(ActiveLearningError):
    """Not enough annotations for retraining"""



class RetrainingError(ActiveLearningError):
    """Model retraining failed"""



# Federated learning exceptions
class FederatedLearningError(HistoCoreError):
    """Base exception for federated learning"""



class ClientConnectionError(FederatedLearningError):
    """Failed to connect to federated client"""



class AggregationError(FederatedLearningError):
    """Model aggregation failed"""



class PrivacyBudgetExceededError(FederatedLearningError):
    """Differential privacy budget exceeded"""



# PACS exceptions
class PACSError(HistoCoreError):
    """Base exception for PACS operations"""



class PACSConnectionError(PACSError):
    """Failed to connect to PACS server"""



class DICOMError(PACSError):
    """DICOM operation failed"""



class StudyNotFoundError(PACSError):
    """DICOM study not found"""



# Streaming exceptions
class StreamingError(HistoCoreError):
    """Base exception for WSI streaming"""



class TileExtractionError(StreamingError):
    """Failed to extract tile from WSI"""



class StreamingConnectionError(StreamingError):
    """Streaming connection lost"""



# Security exceptions
class SecurityError(HistoCoreError):
    """Base exception for security violations"""



class AuthenticationError(SecurityError):
    """Authentication failed"""



class AuthorizationError(SecurityError):
    """User not authorized for operation"""



class IntegrityError(SecurityError):
    """Data integrity check failed"""



class EncryptionError(SecurityError):
    """Encryption/decryption failed"""



# Resource exceptions
class ResourceError(HistoCoreError):
    """Base exception for resource issues"""



class OutOfMemoryError(ResourceError):
    """Insufficient memory for operation"""



class DiskSpaceError(ResourceError):
    """Insufficient disk space"""



class GPUError(ResourceError):
    """GPU operation failed"""



class TimeoutError(ResourceError):
    """Operation timed out"""



# Threading exceptions
class ThreadingError(HistoCoreError):
    """Base exception for threading issues"""



class DeadlockError(ThreadingError):
    """Deadlock detected"""



class ThreadPoolExhaustedError(ThreadingError):
    """Thread pool has no available threads"""

