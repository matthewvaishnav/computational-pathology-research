"""Production configuration management."""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field, validator
from pathlib import Path


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    
    url: str = Field(..., env="DATABASE_URL")
    pool_size: int = Field(10, env="DB_POOL_SIZE")
    max_overflow: int = Field(20, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(30, env="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(3600, env="DB_POOL_RECYCLE")


class RedisConfig(BaseSettings):
    """Redis configuration."""
    
    url: str = Field(..., env="REDIS_URL")
    max_connections: int = Field(100, env="REDIS_MAX_CONNECTIONS")
    retry_on_timeout: bool = Field(True, env="REDIS_RETRY_ON_TIMEOUT")
    socket_timeout: int = Field(5, env="REDIS_SOCKET_TIMEOUT")


class SecurityConfig(BaseSettings):
    """Security configuration."""
    
    secret_key: str = Field(..., env="SECRET_KEY")
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(24, env="JWT_EXPIRATION_HOURS")
    
    # TLS Configuration
    tls_cert_path: str = Field("/app/certs/server-cert.pem", env="TLS_CERT_PATH")
    tls_key_path: str = Field("/app/certs/server-key.pem", env="TLS_KEY_PATH")
    ca_cert_path: str = Field("/app/certs/ca-cert.pem", env="CA_CERT_PATH")
    
    # HIPAA Compliance
    audit_log_enabled: bool = Field(True, env="AUDIT_LOG_ENABLED")
    audit_log_path: str = Field("/app/logs/audit.log", env="AUDIT_LOG_PATH")
    
    # Password Policy
    min_password_length: int = Field(12, env="MIN_PASSWORD_LENGTH")
    require_special_chars: bool = Field(True, env="REQUIRE_SPECIAL_CHARS")
    password_expiry_days: int = Field(90, env="PASSWORD_EXPIRY_DAYS")


class FederatedLearningConfig(BaseSettings):
    """Federated learning configuration."""
    
    # Training Parameters
    max_clients_per_round: int = Field(100, env="MAX_CLIENTS_PER_ROUND")
    min_clients_per_round: int = Field(2, env="MIN_CLIENTS_PER_ROUND")
    round_timeout_seconds: int = Field(3600, env="ROUND_TIMEOUT_SECONDS")  # 1 hour
    
    # Privacy Parameters
    default_epsilon: float = Field(1.0, env="DEFAULT_EPSILON")
    default_delta: float = Field(1e-5, env="DEFAULT_DELTA")
    max_epsilon_per_client: float = Field(10.0, env="MAX_EPSILON_PER_CLIENT")
    
    # Byzantine Robustness
    byzantine_detection_enabled: bool = Field(True, env="BYZANTINE_DETECTION_ENABLED")
    max_byzantine_ratio: float = Field(0.3, env="MAX_BYZANTINE_RATIO")
    
    # Model Management
    checkpoint_interval: int = Field(5, env="CHECKPOINT_INTERVAL")  # Every 5 rounds
    max_checkpoints: int = Field(50, env="MAX_CHECKPOINTS")
    model_compression_enabled: bool = Field(True, env="MODEL_COMPRESSION_ENABLED")


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration."""
    
    metrics_enabled: bool = Field(True, env="METRICS_ENABLED")
    metrics_port: int = Field(8000, env="METRICS_PORT")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")
    log_file_path: str = Field("/app/logs/app.log", env="LOG_FILE_PATH")
    
    # Health Checks
    health_check_interval: int = Field(30, env="HEALTH_CHECK_INTERVAL")
    
    # Sentry (Error Tracking)
    sentry_dsn: Optional[str] = Field(None, env="SENTRY_DSN")
    sentry_environment: str = Field("production", env="SENTRY_ENVIRONMENT")


class ProductionConfig(BaseSettings):
    """Main production configuration."""
    
    # Environment
    environment: str = Field("production", env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    
    # Server Configuration
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8080, env="PORT")
    grpc_port: int = Field(50051, env="GRPC_PORT")
    workers: int = Field(4, env="WORKERS")
    
    # Component Configurations
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    security: SecurityConfig = SecurityConfig()
    federated_learning: FederatedLearningConfig = FederatedLearningConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    
    # Enterprise Features
    ldap_enabled: bool = Field(False, env="LDAP_ENABLED")
    ldap_server: Optional[str] = Field(None, env="LDAP_SERVER")
    ldap_base_dn: Optional[str] = Field(None, env="LDAP_BASE_DN")
    
    # Kubernetes Integration
    kubernetes_enabled: bool = Field(False, env="KUBERNETES_ENABLED")
    kubernetes_namespace: str = Field("federated-learning", env="KUBERNETES_NAMESPACE")
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment setting."""
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global configuration instance
config = ProductionConfig()


def get_config() -> ProductionConfig:
    """Get the global configuration instance."""
    return config


def validate_production_config():
    """Validate production configuration."""
    errors = []
    
    # Check required files exist
    required_files = [
        config.security.tls_cert_path,
        config.security.tls_key_path,
        config.security.ca_cert_path,
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            errors.append(f"Required file not found: {file_path}")
    
    # Check database connection
    if not config.database.url:
        errors.append("DATABASE_URL is required")
    
    # Check Redis connection
    if not config.redis.url:
        errors.append("REDIS_URL is required")
    
    # Check security settings
    if not config.security.secret_key:
        errors.append("SECRET_KEY is required")
    
    if len(config.security.secret_key) < 32:
        errors.append("SECRET_KEY must be at least 32 characters")
    
    # Check FL parameters
    if config.federated_learning.min_clients_per_round > config.federated_learning.max_clients_per_round:
        errors.append("MIN_CLIENTS_PER_ROUND cannot be greater than MAX_CLIENTS_PER_ROUND")
    
    if config.federated_learning.max_byzantine_ratio >= 0.5:
        errors.append("MAX_BYZANTINE_RATIO must be less than 0.5")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    
    return True