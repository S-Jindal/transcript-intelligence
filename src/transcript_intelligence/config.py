from functools import cached_property

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(min_length=1)

    llm_model: str = "gpt-4.1-mini"
    llm_concurrency: int = Field(default=8, ge=1)
    llm_maximum_attempts: int = Field(default=4, ge=1)
    llm_initial_backoff_seconds: float = Field(default=1.0, ge=0)
    llm_timeout_seconds: float = Field(default=90.0, gt=0)

    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_device: str = "auto"
    spacy_model: str = "en_core_web_sm"

    maximum_segment_tokens: int = Field(default=500, ge=1, le=500)
    semantic_similarity_threshold: float = Field(default=0.35, ge=-1, le=1)
    minimum_cluster_size: int = Field(default=5, ge=2)
    centroid_segment_count: int = Field(default=3, ge=2, le=3)
    topic_random_state: int = 42

    classify_utterance_window: int = Field(default=10, ge=1)
    classify_confidence_threshold: float = Field(default=0.7, ge=0, le=1)

    topic_prompt_version: str = "topic-label-v1"
    findings_prompt_version: str = "segment-findings-v1"

    allowlist: str = "Aegis Cloud Security,Aegis Identity"
    account_patterns: str = (
        r"(?i)\b(?:account|customer|case)\s*(?:number|no\.?|id)"
        r"\s*[:#-]?\s*([A-Z0-9-]{4,})\b"
    )

    @field_validator("openai_api_key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("OPENAI_API_KEY cannot be empty")
        return stripped

    @cached_property
    def allowlist_terms(self) -> tuple[str, ...]:
        return tuple(
            term.strip()
            for term in self.allowlist.split(",")
            if term.strip()
        )

    @cached_property
    def account_pattern_list(self) -> tuple[str, ...]:
        return tuple(
            pattern.strip()
            for pattern in self.account_patterns.split("\n")
            if pattern.strip()
        )

    def resolved_embedding_device(self) -> str:
        if self.embedding_device != "auto":
            return self.embedding_device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
