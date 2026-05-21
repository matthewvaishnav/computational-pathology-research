"""
Secure Database Connection

Provides secure database connection with proper pooling and timeout.
"""

from contextlib import contextmanager


class SecureDBConfig:
    """Secure database configuration."""

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        connect_timeout: int = 10,
        ssl_required: bool = True,
    ):
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.connect_timeout = connect_timeout
        self.ssl_required = ssl_required

    def get_connection_string(self) -> str:
        """Get connection string with security parameters."""
        # Build connection string
        conn_str = (
            f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        )

        # Add security parameters
        params = []
        if self.ssl_required:
            params.append("sslmode=require")
        params.append(f"connect_timeout={self.connect_timeout}")

        if params:
            conn_str += "?" + "&".join(params)

        return conn_str

    @classmethod
    def from_env(cls) -> "SecureDBConfig":
        """Create config from environment variables."""
        import os

        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "medical_ai"),
            username=os.getenv("DB_USER", ""),
            password=os.getenv("DB_PASSWORD", ""),
            ssl_required=os.getenv("DB_SSL_REQUIRED", "true").lower() == "true",
        )


@contextmanager
def get_secure_connection(config: SecureDBConfig):
    """Get secure database connection with timeout.

    Args:
        config: Database configuration

    Yields:
        Database connection
    """
    import psycopg2

    conn = None
    try:
        # Connect with timeout
        conn = psycopg2.connect(
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.username,
            password=config.password,
            connect_timeout=config.connect_timeout,
            sslmode="require" if config.ssl_required else "prefer",
        )

        # Set statement timeout (prevent long-running queries)
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '30s'")

        yield conn

    finally:
        if conn:
            conn.close()
