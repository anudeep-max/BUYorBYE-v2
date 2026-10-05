"""Configuration management for the application."""

import json
import logging
from typing import Optional

from pydantic_settings import BaseSettings

_logger = logging.getLogger(__name__)

_DEFAULT_SECRET_KEY = "your-secret-key-change-in-production"


def _load_aws_secrets(secret_name: str, region: str) -> dict:
    """Load secrets from AWS Secrets Manager."""
    try:
        import boto3

        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
        secret_string = response.get("SecretString", "{}")
        return json.loads(secret_string)
    except ImportError:
        _logger.warning(
            "boto3 not installed – skipping AWS Secrets Manager load."
        )
        return {}
    except Exception as exc:
        _logger.warning(
            "Could not load AWS secret '%s': %s",
            secret_name,
            exc,
        )
        return {}


class Settings(BaseSettings):
    """Application settings."""

    # ------------------------------------------------------------------
    # API Keys
    # ------------------------------------------------------------------
    openai_api_key: str = ""
    anthropic_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None

    tavily_api_key: Optional[str] = None
    serper_api_key: Optional[str] = None
    etsy_api_key: Optional[str] = None

    # Multi-source product APIs
    amazon_api_key: Optional[str] = None
    amazon_secret_key: Optional[str] = None
    amazon_associate_tag: Optional[str] = None
    ebay_api_key: Optional[str] = None
    walmart_api_key: Optional[str] = None
    bestbuy_api_key: Optional[str] = None
    pricegrabber_api_key: Optional[str] = None
    shopzilla_api_key: Optional[str] = None

    # Coupon / Promo APIs
    honey_api_key: Optional[str] = None
    retailmenot_api_key: Optional[str] = None
    couponfollow_api_key: Optional[str] = None

    # Currency conversion
    exchangerate_api_key: Optional[str] = None

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: str = "sqlite:///./shopping_assistant.db"

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = 0.3

    # ------------------------------------------------------------------
    # API Configuration
    # ------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 3565

    secret_key: str = _DEFAULT_SECRET_KEY
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    api_key: Optional[str] = None

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    rate_limit_per_minute: int = 60

    # ------------------------------------------------------------------
    # Redis / Cache
    # ------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = False

    cache_llm_response_ttl: int = 3600
    cache_product_search_ttl: int = 1800
    cache_session_ttl: int = 3600
    cache_embedding_ttl: int = 86400

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = "INFO"
    log_format: str = "text"
    log_file: str = "logs/app.log"

    # ------------------------------------------------------------------
    # Langfuse
    # ------------------------------------------------------------------
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_project_name: str = "shopping-assistant"
    langfuse_enabled: bool = False

    # ------------------------------------------------------------------
    # DeepEval
    # ------------------------------------------------------------------
    deepeval_api_key: Optional[str] = None
    deepeval_enabled: bool = False

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    environment: str = "development"
    production_mode: bool = False

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    debug_prompts: bool = False

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    cors_origins: str = "*"

    # ------------------------------------------------------------------
    # Model routing
    # ------------------------------------------------------------------
    enable_model_routing: bool = False

    # ------------------------------------------------------------------
    # Conversation context
    # ------------------------------------------------------------------
    max_history_exchanges: int = 10
    recent_exchanges_full: int = 5
    older_exchanges_truncate: int = 300

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"
    use_openai_embeddings: bool = False

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    semantic_only_retrieval: bool = True

    # ------------------------------------------------------------------
    # Product aggregation
    # ------------------------------------------------------------------
    product_source_priority: str = (
        "price_comparison,direct_retailers,serper"
    )

    enable_price_comparison: bool = True
    enable_price_history: bool = True
    enable_coupon_integration: bool = True
    max_retailers_per_product: int = 5

    # ------------------------------------------------------------------
    # AWS
    # ------------------------------------------------------------------
    aws_region: str = "us-east-1"
    aws_secrets_name: Optional[str] = None

    xray_enabled: bool = False

    cloudwatch_enabled: bool = False
    cloudwatch_namespace: str = "ShoppingAssistant/Application"

    bedrock_enabled: bool = False

    # ------------------------------------------------------------------
    # Pydantic configuration
    # ------------------------------------------------------------------
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # --------------------------------------------------------------
        # AWS Secrets Manager
        # --------------------------------------------------------------
        if self.aws_secrets_name:
            aws_secrets = _load_aws_secrets(
                self.aws_secrets_name,
                self.aws_region,
            )

            for field_name, value in aws_secrets.items():
                lower_field = field_name.lower()

                if (
                    lower_field in self.model_fields
                    and value is not None
                ):
                    object.__setattr__(
                        self,
                        lower_field,
                        value,
                    )

        # --------------------------------------------------------------
        # Production mode
        # --------------------------------------------------------------
        self.production_mode = (
            self.environment.lower()
            in ("production", "prod")
        )

        if self.production_mode:
            if self.log_level == "INFO":
                self.log_level = "WARNING"

            if self.secret_key == _DEFAULT_SECRET_KEY:
                raise ValueError(
                    "SECRET_KEY is set to the default placeholder value. "
                    "Set a strong, random SECRET_KEY in your environment "
                    "before starting the application in production."
                )


settings = Settings()
